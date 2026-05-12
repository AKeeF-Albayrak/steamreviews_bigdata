from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lower, trim, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


KAFKA_BOOTSTRAP_SERVERS = "steam_kafka:9092"
KAFKA_TOPIC = "steam_reviews"

BRONZE_PATH = "/app/delta/bronze/steam_reviews_raw"
SILVER_PATH = "/app/delta/silver/steam_reviews_clean"

BRONZE_CHECKPOINT = "/app/checkpoints/bronze_steam_reviews"


spark = (
    SparkSession.builder
    .appName("SteamReviewsStructuredStreaming")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


review_schema = StructType([
    StructField("event_id", IntegerType(), True),
    StructField("event_type", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("app_id", IntegerType(), True),
    StructField("app_name", StringType(), True),
    StructField("review_text", StringType(), True),
    StructField("review_score", IntegerType(), True),
    StructField("review_votes", IntegerType(), True),
    StructField("sentiment", StringType(), True),
])


print("Kafka -> Bronze Delta streaming başlatılıyor...")

raw_kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", "50000")
    .load()
)

bronze_df = (
    raw_kafka_df
    .selectExpr(
        "CAST(key AS STRING) AS kafka_key",
        "CAST(value AS STRING) AS raw_json",
        "topic",
        "partition",
        "offset",
        "timestamp AS kafka_timestamp"
    )
    .withColumn("ingest_time", current_timestamp())
)

bronze_query = (
    bronze_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", BRONZE_CHECKPOINT)
    .trigger(availableNow=True)
    .start(BRONZE_PATH)
)

bronze_query.awaitTermination()

print("Bronze Delta yazımı tamamlandı.")
print("Bronze Delta -> Silver Delta batch temizleme başlatılıyor...")


bronze_static_df = spark.read.format("delta").load(BRONZE_PATH)

parsed_df = (
    bronze_static_df
    .select(from_json(col("raw_json"), review_schema).alias("data"))
    .select("data.*")
)

silver_df = (
    parsed_df
    .withColumn("event_time", to_timestamp(col("timestamp")))
    .withColumn("event_type", trim(lower(col("event_type"))))
    .withColumn("sentiment", trim(lower(col("sentiment"))))
    .filter(col("event_id").isNotNull())
    .filter(col("app_id").isNotNull())
    .filter(col("app_name").isNotNull())
    .filter(col("review_score").isNotNull())
    .filter(col("sentiment").isin("positive", "negative"))
    .repartition(8, "app_id")
)

(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .save(SILVER_PATH)
)

print("Silver Delta yazımı tamamlandı.")
print(f"Kafka topic: {KAFKA_TOPIC}")
print(f"Bronze Delta path: {BRONZE_PATH}")
print(f"Silver Delta path: {SILVER_PATH}")
print("Spark Structured Streaming tamamlandı.")

spark.stop()