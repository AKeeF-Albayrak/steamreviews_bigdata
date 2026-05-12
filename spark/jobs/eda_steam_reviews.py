import os

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    date_trunc,
    desc,
    length,
    when
)


SILVER_PATH = "/app/delta/silver/steam_reviews_clean"
EDA_OUTPUT_PATH = "/app/reports/eda_outputs"
FIGURE_PATH = "/app/reports/figures"

os.makedirs(EDA_OUTPUT_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)


spark = (
    SparkSession.builder
    .appName("SteamReviewsEDA")
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.format("delta").load(SILVER_PATH)

print("Silver Delta verisi okundu.")
print("Schema:")
df.printSchema()

total_rows = df.count()
unique_apps = df.select(countDistinct("app_id")).collect()[0][0]
unique_sentiments = df.select(countDistinct("sentiment")).collect()[0][0]

summary_rows = [
    ("total_rows", total_rows),
    ("unique_apps", unique_apps),
    ("unique_sentiments", unique_sentiments),
]

summary_df = spark.createDataFrame(summary_rows, ["metric", "value"])
summary_df.toPandas().to_csv(f"{EDA_OUTPUT_PATH}/basic_summary.csv", index=False)

missing_exprs = [
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]
missing_df = df.select(missing_exprs)
missing_df.toPandas().to_csv(f"{EDA_OUTPUT_PATH}/missing_values.csv", index=False)

sentiment_df = (
    df.groupBy("sentiment")
    .count()
    .orderBy(desc("count"))
)
sentiment_df.toPandas().to_csv(f"{EDA_OUTPUT_PATH}/sentiment_distribution.csv", index=False)

score_df = (
    df.groupBy("review_score")
    .count()
    .orderBy("review_score")
)
score_df.toPandas().to_csv(f"{EDA_OUTPUT_PATH}/review_score_distribution.csv", index=False)

top_apps_df = (
    df.groupBy("app_id", "app_name")
    .agg(
        count("*").alias("review_count"),
        avg("review_votes").alias("avg_review_votes")
    )
    .orderBy(desc("review_count"))
)
top_apps_df.limit(20).toPandas().to_csv(f"{EDA_OUTPUT_PATH}/top_apps.csv", index=False)

hourly_df = (
    df.withColumn("hour", date_trunc("hour", col("event_time")))
    .groupBy("hour")
    .count()
    .orderBy("hour")
)
hourly_df.toPandas().to_csv(f"{EDA_OUTPUT_PATH}/hourly_trend.csv", index=False)

text_length_summary_df = (
    df.withColumn("review_text_length", length(col("review_text")))
    .groupBy("sentiment")
    .agg(
        count("*").alias("review_count"),
        avg("review_text_length").alias("avg_review_text_length")
    )
)

text_length_summary_df.toPandas().to_csv(
    f"{EDA_OUTPUT_PATH}/review_text_length_summary.csv",
    index=False
)

# Grafik 1: Sentiment dağılımı
sentiment_pd = sentiment_df.toPandas()
plt.figure(figsize=(8, 5))
plt.bar(sentiment_pd["sentiment"], sentiment_pd["count"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Review Count")
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/sentiment_distribution.png")
plt.close()

# Grafik 2: Review score dağılımı
score_pd = score_df.toPandas()
plt.figure(figsize=(8, 5))
plt.bar(score_pd["review_score"].astype(str), score_pd["count"])
plt.title("Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Review Count")
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/review_score_distribution.png")
plt.close()

# Grafik 3: En çok yorum alan oyunlar
top_apps_pd = top_apps_df.limit(10).toPandas()
plt.figure(figsize=(10, 6))
plt.barh(top_apps_pd["app_name"], top_apps_pd["review_count"])
plt.title("Top 10 Apps by Review Count")
plt.xlabel("Review Count")
plt.ylabel("App Name")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/top_10_apps.png")
plt.close()

# Grafik 4: Saatlik trend
hourly_pd = hourly_df.toPandas()
plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["count"], marker="o")
plt.title("Hourly Review Stream Trend")
plt.xlabel("Hour")
plt.ylabel("Review Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/hourly_trend.png")
plt.close()

print("EDA tamamlandı.")
print(f"CSV çıktıları: {EDA_OUTPUT_PATH}")
print(f"Grafikler: {FIGURE_PATH}")

spark.stop()