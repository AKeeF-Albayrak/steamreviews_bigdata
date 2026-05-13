import os

from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType


SILVER_PATH = "/app/delta/silver/steam_reviews_clean"
MODEL_PATH = "/app/ml/models/sentiment_lr_model"
ML_OUTPUT_PATH = "/app/reports/ml_outputs"

# train_sentiment_model.py ile aynı parametreler — aynı test split'ini üretmek için
SAMPLE_PER_CLASS = 100000
SEED = 42

os.makedirs(ML_OUTPUT_PATH, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("ExportBestPredictions")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Silver Delta verisi okunuyor...")
df = spark.read.format("delta").load(SILVER_PATH)

model_df = (
    df.select("event_id", "review_text", "sentiment")
    .filter(col("review_text").isNotNull())
    .filter(col("sentiment").isin("positive", "negative"))
    .withColumn("label_text", col("sentiment"))
)

# train_sentiment_model.py ile aynı dengeli örneklem
negative_df = model_df.filter(col("label_text") == "negative").limit(SAMPLE_PER_CLASS)
positive_df = model_df.filter(col("label_text") == "positive").limit(SAMPLE_PER_CLASS)
balanced_df = negative_df.unionByName(positive_df)

# train_sentiment_model.py ile aynı split (seed=42)
_, test_df = balanced_df.randomSplit([0.8, 0.2], seed=SEED)

print("Model yükleniyor...")
model = PipelineModel.load(MODEL_PATH)

print("Test tahmini yapılıyor...")
predictions = model.transform(test_df)

# Sayısal tahmini metne çevir (stages[0] = StringIndexer, alphabetAsc: negative=0, positive=1)
prediction_converter = IndexToString(
    inputCol="prediction",
    outputCol="prediction_text",
    labels=model.stages[0].labels
)
predictions = prediction_converter.transform(predictions)

# probability vektöründen float değerleri çıkar: [P(negative), P(positive)]
prob_neg_udf = udf(lambda v: float(v[0]), FloatType())
prob_pos_udf = udf(lambda v: float(v[1]), FloatType())

output_df = (
    predictions
    .withColumn("probability_negative", prob_neg_udf(col("probability")))
    .withColumn("probability_positive", prob_pos_udf(col("probability")))
    .select(
        "event_id",
        "review_text",
        "label_text",
        "prediction_text",
        "probability_negative",
        "probability_positive",
    )
)

output_path = f"{ML_OUTPUT_PATH}/best_model_predictions.csv"
output_df.toPandas().to_csv(output_path, index=False)

count = output_df.count()
print(f"Tamamlandi. Toplam tahmin: {count}")
print(f"Cikti: {output_path}")

spark.stop()
