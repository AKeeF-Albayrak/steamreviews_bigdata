import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, StringIndexer, Tokenizer, IndexToString
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count


SILVER_PATH = "/app/delta/silver/steam_reviews_clean"
ML_OUTPUT_PATH = "/app/reports/ml_outputs"
MODEL_PATH = "/app/ml/models/sentiment_lr_model"

os.makedirs(ML_OUTPUT_PATH, exist_ok=True)
os.makedirs("/app/ml/models", exist_ok=True)


spark = (
    SparkSession.builder
    .appName("SteamReviewsSentimentModel")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Silver Delta verisi okunuyor...")

df = spark.read.format("delta").load(SILVER_PATH)

model_df = (
    df.select("review_text", "sentiment")
    .filter(col("review_text").isNotNull())
    .filter(col("sentiment").isin("positive", "negative"))
    .withColumn("label_text", col("sentiment"))
)

print("Model verisi hazırlandı.")
print("Sınıf dağılımı:")

class_distribution_df = (
    model_df.groupBy("label_text")
    .count()
    .orderBy("label_text")
)

class_distribution_df.show()

class_distribution_df.toPandas().to_csv(
    f"{ML_OUTPUT_PATH}/class_distribution.csv",
    index=False
)

# Verinin çok büyük olması nedeniyle lokal geliştirme ortamında dengeli örneklem kullanıyoruz.
# Negatif sınıf yaklaşık 932K olduğu için her sınıftan maksimum 900K alınır.
# hızlandırmak için 250kya düşürdük
negative_df = model_df.filter(col("label_text") == "negative").limit(100000)
positive_df = model_df.filter(col("label_text") == "positive").limit(100000)

balanced_df = negative_df.unionByName(positive_df)

print("Dengeli eğitim verisi oluşturuldu:")
balanced_df.groupBy("label_text").count().show()

train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)

label_indexer = StringIndexer(
    inputCol="label_text",
    outputCol="label",
    handleInvalid="skip",
    stringOrderType="alphabetAsc"
)

tokenizer = Tokenizer(
    inputCol="review_text",
    outputCol="words"
)

stopwords_remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

hashing_tf = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_features",
    numFeatures=20000
)

idf = IDF(
    inputCol="raw_features",
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        tokenizer,
        stopwords_remover,
        hashing_tf,
        idf,
        lr
    ]
)

print("Model eğitimi başlatılıyor...")

model = pipeline.fit(train_df)

print("Model eğitimi tamamlandı.")
print("Test tahminleri alınıyor...")

predictions = model.transform(test_df)

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
auc = auc_evaluator.evaluate(predictions)

metrics_rows = [
    ("accuracy", float(accuracy)),
    ("f1_score", float(f1_score)),
    ("weighted_precision", float(precision)),
    ("weighted_recall", float(recall)),
    ("area_under_roc", float(auc)),
]

metrics_df = spark.createDataFrame(metrics_rows, ["metric", "value"])
metrics_df.toPandas().to_csv(f"{ML_OUTPUT_PATH}/model_metrics.csv", index=False)

prediction_converter = IndexToString(
    inputCol="prediction",
    outputCol="prediction_text",
    labels=model.stages[0].labels
)

predictions_with_text = prediction_converter.transform(predictions)

confusion_matrix_df = (
    predictions_with_text
    .groupBy("label_text", "prediction_text")
    .agg(count("*").alias("count"))
    .orderBy("label_text", "prediction_text")
)

confusion_matrix_df.toPandas().to_csv(
    f"{ML_OUTPUT_PATH}/confusion_matrix.csv",
    index=False
)

prediction_distribution_df = (
    predictions_with_text
    .groupBy("prediction_text")
    .count()
    .orderBy("prediction_text")
)

prediction_distribution_df.toPandas().to_csv(
    f"{ML_OUTPUT_PATH}/prediction_distribution.csv",
    index=False
)

print("Model metrikleri:")
metrics_df.show(truncate=False)

print("Confusion matrix CSV olarak kaydedildi.")

print("Model kaydediliyor...")

model.write().overwrite().save(MODEL_PATH)

print("Modelleme tamamlandı.")
print(f"ML çıktı klasörü: {ML_OUTPUT_PATH}")
print(f"Model klasörü: {MODEL_PATH}")

spark.stop()
