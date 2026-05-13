import os
import shutil
import json

import mlflow
import pandas as pd

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    NaiveBayes
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    CountVectorizer,
    IDF,
    StopWordsRemover,
    StringIndexer,
    Tokenizer,
    IndexToString
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count


SILVER_PATH = "/app/delta/silver/steam_reviews_clean"

ML_OUTPUT_PATH = "/app/reports/ml_outputs"
MLFLOW_PATH = "/app/reports/mlruns"

EXISTING_LR_MODEL_PATH = "/app/ml/models/sentiment_lr_model"
MODEL_BASE_PATH = "/app/ml/models"

SAMPLE_PER_CLASS = 25000
VOCAB_SIZE = 5000
MIN_DF = 5
SEED = 42

os.makedirs(ML_OUTPUT_PATH, exist_ok=True)
os.makedirs(MLFLOW_PATH, exist_ok=True)
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_PATH}")
mlflow.set_experiment("Steam Reviews Multi Model Comparison")


spark = (
    SparkSession.builder
    .appName("SteamReviewsMultiModelMLflow")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def safe_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def build_text_pipeline(classifier):
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

    count_vectorizer = CountVectorizer(
        inputCol="filtered_words",
        outputCol="tf_features",
        vocabSize=VOCAB_SIZE,
        minDF=MIN_DF
    )

    idf = IDF(
        inputCol="tf_features",
        outputCol="features"
    )

    return Pipeline(
        stages=[
            label_indexer,
            tokenizer,
            stopwords_remover,
            count_vectorizer,
            idf,
            classifier
        ]
    )


def evaluate_predictions(predictions):
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

    return {
        "accuracy": float(accuracy_evaluator.evaluate(predictions)),
        "f1_score": float(f1_evaluator.evaluate(predictions)),
        "weighted_precision": float(precision_evaluator.evaluate(predictions)),
        "weighted_recall": float(recall_evaluator.evaluate(predictions)),
        "area_under_roc": float(auc_evaluator.evaluate(predictions)),
    }


def save_confusion_matrix(predictions, fitted_model, model_name):
    prediction_converter = IndexToString(
        inputCol="prediction",
        outputCol="prediction_text",
        labels=fitted_model.stages[0].labels
    )

    predictions_with_text = prediction_converter.transform(predictions)

    confusion_df = (
        predictions_with_text
        .groupBy("label_text", "prediction_text")
        .agg(count("*").alias("count"))
        .orderBy("label_text", "prediction_text")
    )

    output_file = f"{ML_OUTPUT_PATH}/confusion_matrix_{safe_name(model_name)}.csv"
    confusion_df.toPandas().to_csv(output_file, index=False)

    return output_file


def get_vocabulary_from_model(fitted_model):
    for stage in fitted_model.stages:
        if hasattr(stage, "vocabulary"):
            return stage.vocabulary
    return None


def save_feature_importance(fitted_model, model_name):
    rows = []
    vocabulary = get_vocabulary_from_model(fitted_model)
    classifier_model = fitted_model.stages[-1]

    # Tree-based models: Decision Tree, Random Forest, GBT
    if hasattr(classifier_model, "featureImportances"):
        importances = classifier_model.featureImportances.toArray()
        top_indices = importances.argsort()[-30:][::-1]

        for idx in top_indices:
            feature_name = vocabulary[idx] if vocabulary and idx < len(vocabulary) else f"feature_{idx}"
            rows.append({
                "model_name": model_name,
                "feature": feature_name,
                "importance": float(importances[idx])
            })

    # Logistic Regression
    elif hasattr(classifier_model, "coefficients"):
        coefficients = classifier_model.coefficients.toArray()
        top_indices = abs(coefficients).argsort()[-30:][::-1]

        for idx in top_indices:
            feature_name = vocabulary[idx] if vocabulary and idx < len(vocabulary) else f"hash_feature_{idx}"
            rows.append({
                "model_name": model_name,
                "feature": feature_name,
                "importance": float(coefficients[idx])
            })

    # Naive Bayes
    elif hasattr(classifier_model, "theta"):
        theta = classifier_model.theta.toArray()

        if theta.shape[0] >= 2:
            importances = abs(theta[1] - theta[0])
            top_indices = importances.argsort()[-30:][::-1]

            for idx in top_indices:
                feature_name = vocabulary[idx] if vocabulary and idx < len(vocabulary) else f"feature_{idx}"
                rows.append({
                    "model_name": model_name,
                    "feature": feature_name,
                    "importance": float(importances[idx])
                })

    if not rows:
        rows.append({
            "model_name": model_name,
            "feature": "not_available",
            "importance": 0.0
        })

    output_file = f"{ML_OUTPUT_PATH}/feature_importance_{safe_name(model_name)}.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)

    return output_file, rows


def save_model_artifact(fitted_model, model_name):
    model_dir = f"{MODEL_BASE_PATH}/{safe_name(model_name)}"

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    fitted_model.write().overwrite().save(model_dir)

    return model_dir


def log_model_run(model_name, params, metrics, confusion_file, feature_file, model_dir):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("sample_per_class", SAMPLE_PER_CLASS)
        mlflow.log_param("vocab_size", VOCAB_SIZE)
        mlflow.log_param("min_df", MIN_DF)

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(confusion_file, artifact_path="confusion_matrix")
        mlflow.log_artifact(feature_file, artifact_path="feature_importance")
        mlflow.log_artifacts(model_dir, artifact_path="model")


print("Silver Delta verisi okunuyor...")

df = spark.read.format("delta").load(SILVER_PATH)

model_df = (
    df.select("review_text", "sentiment")
    .filter(col("review_text").isNotNull())
    .filter(col("sentiment").isin("positive", "negative"))
    .withColumn("label_text", col("sentiment"))
)

print("Sınıf dağılımı:")
model_df.groupBy("label_text").count().show()

negative_df = model_df.filter(col("label_text") == "negative").limit(SAMPLE_PER_CLASS)
positive_df = model_df.filter(col("label_text") == "positive").limit(SAMPLE_PER_CLASS)

balanced_df = negative_df.unionByName(positive_df).repartition(4)

print("6. adım için dengeli veri oluşturuldu:")
balanced_df.groupBy("label_text").count().show()

train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=SEED)

comparison_rows = []
all_feature_rows = []

print("Mevcut Logistic Regression modeli yükleniyor, tekrar eğitilmeyecek...")

existing_lr_model = PipelineModel.load(EXISTING_LR_MODEL_PATH)
lr_predictions = existing_lr_model.transform(test_df)

lr_metrics = evaluate_predictions(lr_predictions)
lr_confusion_file = save_confusion_matrix(
    lr_predictions,
    existing_lr_model,
    "Logistic Regression Existing"
)
lr_feature_file, lr_feature_rows = save_feature_importance(
    existing_lr_model,
    "Logistic Regression Existing"
)

log_model_run(
    model_name="Logistic Regression Existing",
    params={
        "source": "loaded_existing_model",
        "retrained": "false"
    },
    metrics=lr_metrics,
    confusion_file=lr_confusion_file,
    feature_file=lr_feature_file,
    model_dir=EXISTING_LR_MODEL_PATH
)

comparison_rows.append({
    "model_name": "Logistic Regression Existing",
    **lr_metrics
})

all_feature_rows.extend(lr_feature_rows)

models = [
    (
        "Decision Tree Classifier",
        DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            seed=SEED
        ),
        {
            "maxDepth": 5
        }
    ),
    (
        "Random Forest Classifier",
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=20,
            maxDepth=5,
            seed=SEED
        ),
        {
            "numTrees": 20,
            "maxDepth": 5
        }
    ),
    (
        "Gradient Boosted Trees Classifier",
        GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=20,
            maxDepth=5,
            seed=SEED
        ),
        {
            "maxIter": 20,
            "maxDepth": 5
        }
    ),
    (
        "Naive Bayes",
        NaiveBayes(
            featuresCol="features",
            labelCol="label",
            smoothing=1.0,
            modelType="multinomial"
        ),
        {
            "smoothing": 1.0,
            "modelType": "multinomial"
        }
    ),
]

for model_name, classifier, params in models:
    print(f"{model_name} eğitimi başlatılıyor...")

    pipeline = build_text_pipeline(classifier)
    fitted_model = pipeline.fit(train_df)

    print(f"{model_name} eğitimi tamamlandı. Test tahminleri alınıyor...")

    predictions = fitted_model.transform(test_df)

    metrics = evaluate_predictions(predictions)
    confusion_file = save_confusion_matrix(predictions, fitted_model, model_name)
    feature_file, feature_rows = save_feature_importance(fitted_model, model_name)
    model_dir = save_model_artifact(fitted_model, model_name)

    log_model_run(
        model_name=model_name,
        params=params,
        metrics=metrics,
        confusion_file=confusion_file,
        feature_file=feature_file,
        model_dir=model_dir
    )

    comparison_rows.append({
        "model_name": model_name,
        **metrics
    })

    all_feature_rows.extend(feature_rows)

    print(f"{model_name} tamamlandı.")
    print(json.dumps(metrics, indent=2))


comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values(by="f1_score", ascending=False)
comparison_df.to_csv(f"{ML_OUTPUT_PATH}/model_comparison.csv", index=False)

feature_importance_df = pd.DataFrame(all_feature_rows)
feature_importance_df.to_csv(f"{ML_OUTPUT_PATH}/feature_importance_all_models.csv", index=False)

best_model = comparison_df.iloc[0].to_dict()

with open(f"{ML_OUTPUT_PATH}/best_model_summary.txt", "w", encoding="utf-8") as f:
    f.write("Best Model Summary\n")
    f.write("==================\n")
    for key, value in best_model.items():
        f.write(f"{key}: {value}\n")

print("6. adım tamamlandı.")
print("Model karşılaştırma çıktısı:", f"{ML_OUTPUT_PATH}/model_comparison.csv")
print("Feature importance çıktısı:", f"{ML_OUTPUT_PATH}/feature_importance_all_models.csv")
print("MLflow kayıt yolu:", MLFLOW_PATH)
print("En iyi model:")
print(best_model)

spark.stop()
