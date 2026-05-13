# Steam Reviews Big Data Projesi - Geliştirici Yönergesi

Bu proje, Steam oyun yorumları veri seti üzerinden uçtan uca büyük veri pipeline'ı kurmayı amaçlar.

Proje kapsamında şu ana kadar tamamlanan kısımlar:

- Docker ortamı hazırlandı.
- Zookeeper servisi eklendi.
- Kafka broker servisi eklendi.
- Python Kafka Producer yazıldı.
- Producer, `steam_reviews.csv` dosyasından satır okuyup Kafka'ya JSON mesaj gönderecek hale getirildi.
- Kafka topic adı: `steam_reviews`
- Producer başarıyla test edildi.
- Kafka consumer ile topic içindeki mesajların okunabildiği doğrulandı.
- Spark Structured Streaming ile Kafka verisi Delta Lake katmanlarına aktarıldı.
- Silver Delta katmanı üzerinden EDA çıktıları üretildi.
- Logistic Regression sentiment modeli eğitildi.
- Beş farklı sınıflandırma modeli karşılaştırıldı ve MLflow ile takip edildi.
- Streamlit tabanlı çok sayfalı interaktif dashboard oluşturuldu ve Docker servisi olarak çalıştırıldı.

---

## 1. Proje Klasör Yapısı

Proje ana dizini:

```text
steamreviews_bigdata/
├── data/
│   ├── raw/
│   ├── sample/
│   ├── processed/
│   ├── delta/
│   └── checkpoints/
├── producer/
│   ├── producer.py
│   ├── Dockerfile
│   └── requirements.txt
├── spark/
│   ├── Dockerfile
│   ├── jobs/
│   │   ├── stream_steam_reviews_to_delta.py
│   │   ├── eda_steam_reviews.py
│   │   ├── train_sentiment_model.py
│   │   └── train_compare_models_mlflow.py
│   └── utils/
├── notebooks/
│   └── check_dataset.py
├── ml/
│   └── models/
├── reports/
│   ├── eda_outputs/
│   ├── figures/
│   ├── ml_outputs/
│   └── mlruns/
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   ├── lib/
│   │   ├── io.py
│   │   └── viz.py
│   └── pages/
│       ├── 01_EDA.py
│       ├── 02_Features.py
│       ├── 03_Model_Comparison.py
│       └── 04_Best_Model.py
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Adım 3 - Spark Structured Streaming ve Delta Lake

Bu adımda Kafka üzerinde bulunan `steam_reviews` topic'i Spark Structured Streaming ile okunmuştur.

Spark tarafında iki Delta Lake katmanı oluşturulmuştur:

- Bronze Layer: Kafka'dan gelen ham JSON mesajları saklar.
- Silver Layer: JSON parse edilmiş, temizlenmiş ve kullanılabilir hale getirilmiş Steam review verilerini saklar.

Oluşan Delta yolları:

```text
data/delta/bronze/steam_reviews_raw
data/delta/silver/steam_reviews_clean
```

Spark streaming job dosyası:

```text
spark/jobs/stream_steam_reviews_to_delta.py
```

Çalıştırma komutu:

```bash
docker compose up -d --build spark
```

Log takibi:

```bash
docker logs -f steam_spark_streaming
```

Bu job Kafka'daki mevcut veriyi okuyup Bronze Delta katmanına yazar, ardından Bronze Delta verisini temizleyerek Silver Delta katmanını oluşturur. İşlem tamamlandığında otomatik olarak kapanır.

---

## Adım 4 - Keşifsel Veri Analizi (EDA)

Bu adımda Silver Delta katmanındaki temizlenmiş veri okunarak keşifsel veri analizi yapılmıştır.

EDA dosyası:

```text
spark/jobs/eda_steam_reviews.py
```

Çalıştırma komutu:

```bash
docker compose run --rm spark /opt/spark/bin/spark-submit --driver-memory 4g --conf spark.driver.maxResultSize=2g --packages io.delta:delta-spark_2.12:3.2.0 /app/jobs/eda_steam_reviews.py
```

EDA sonucunda aşağıdaki CSV analiz çıktıları oluşturulur:

```text
reports/eda_outputs/basic_summary.csv
reports/eda_outputs/missing_values.csv
reports/eda_outputs/sentiment_distribution.csv
reports/eda_outputs/review_score_distribution.csv
reports/eda_outputs/top_apps.csv
reports/eda_outputs/hourly_trend.csv
reports/eda_outputs/review_text_length_summary.csv
```

Aşağıdaki grafikler oluşturulur:

```text
reports/figures/sentiment_distribution.png
reports/figures/review_score_distribution.png
reports/figures/top_10_apps.png
reports/figures/hourly_trend.png
```

Not: `data/raw`, `data/delta` ve `data/checkpoints` klasörleri büyük veri ve çalışma çıktıları içerdiği için GitHub'a gönderilmez. Bu klasörler `.gitignore` içinde tutulur.

---

## Adım 5 - Logistic Regression Sentiment Modeli

Bu adımda Silver Delta katmanındaki temizlenmiş Steam yorumları kullanılarak pozitif/negatif duygu analizi için Logistic Regression modeli eğitilmiştir.

Kullanılan veri:

```text
data/delta/silver/steam_reviews_clean
```

Modelleme dosyası:

```text
spark/jobs/train_sentiment_model.py
```

Çalıştırma komutu:

```bash
docker compose run --rm --no-deps spark /opt/spark/bin/spark-submit --driver-memory 4g --conf spark.driver.maxResultSize=2g --packages io.delta:delta-spark_2.12:3.2.0 /app/jobs/train_sentiment_model.py
```

Üretilen çıktılar:

```text
reports/ml_outputs/model_metrics.csv
reports/ml_outputs/confusion_matrix.csv
reports/ml_outputs/prediction_distribution.csv
ml/models/sentiment_lr_model
```

Bu adımda Logistic Regression modeli için Accuracy, F1-score, Precision, Recall, AUC-ROC ve Confusion Matrix hesaplanmıştır.

---

## Adım 6 - Çoklu Model Karşılaştırma ve MLflow Takibi

Bu adımda duygu analizi problemi için beş farklı sınıflandırma modeli karşılaştırılmıştır:

```text
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosted Trees Classifier
5. Naive Bayes
```

Model karşılaştırma dosyası:

```text
spark/jobs/train_compare_models_mlflow.py
```

Çalıştırma komutu:

```bash
docker compose run --rm --no-deps spark /opt/spark/bin/spark-submit --driver-memory 4g --conf spark.driver.maxResultSize=2g --packages io.delta:delta-spark_2.12:3.2.0 /app/jobs/train_compare_models_mlflow.py
```

Üretilen değerlendirme çıktıları:

```text
reports/ml_outputs/model_comparison.csv
reports/ml_outputs/best_model_summary.txt
reports/ml_outputs/feature_importance_all_models.csv
reports/ml_outputs/confusion_matrix_logistic_regression_existing.csv
reports/ml_outputs/confusion_matrix_decision_tree_classifier.csv
reports/ml_outputs/confusion_matrix_random_forest_classifier.csv
reports/ml_outputs/confusion_matrix_gradient_boosted_trees_classifier.csv
reports/ml_outputs/confusion_matrix_naive_bayes.csv
```

Feature Importance çıktıları:

```text
reports/ml_outputs/feature_importance_logistic_regression_existing.csv
reports/ml_outputs/feature_importance_decision_tree_classifier.csv
reports/ml_outputs/feature_importance_random_forest_classifier.csv
reports/ml_outputs/feature_importance_gradient_boosted_trees_classifier.csv
reports/ml_outputs/feature_importance_naive_bayes.csv
reports/ml_outputs/feature_importance_all_models.csv
```

MLflow deney kayıtları:

```text
reports/mlruns/
```

Kaydedilen modeller:

```text
ml/models/
```

Bu adımda her model için Accuracy, F1-score, Precision, Recall, AUC-ROC ve Confusion Matrix hesaplanmıştır. Ayrıca Feature Importance analizi yapılmış ve deneyler MLflow ile loglanmıştır.

En iyi model sonucu:

```text
Logistic Regression Existing
```

---

## Adım 7 - Streamlit Dashboard

Bu adımda EDA bulgularını ve model sonuçlarını görselleştiren çok sayfalı interaktif bir Streamlit dashboard oluşturulmuştur. Dashboard, Docker servisi olarak `docker-compose.yml` içine eklenmiş ve `http://localhost:8501` adresinden erişilebilir.

Dashboard dosyaları:

```text
dashboard/
├── Dockerfile
├── requirements.txt
├── app.py                        ← Landing page (pipeline özeti, metrikler)
├── lib/
│   ├── io.py                     ← CSV okuma ve Silver Delta okuma fonksiyonları
│   └── viz.py                    ← Plotly grafik fonksiyonları
└── pages/
    ├── 01_EDA.py                 ← Keşifsel Veri Analizi (10 bölüm)
    ├── 02_Features.py            ← Feature Engineering (TF-IDF pipeline)
    ├── 03_Model_Comparison.py    ← 5 Model karşılaştırması
    └── 04_Best_Model.py          ← En iyi model derinlemesine analizi
```

ROC Curve için test seti tahminlerini dışa aktaran Spark job:

```text
spark/jobs/export_best_predictions.py
```

Bu job'ı çalıştırmak için (Silver Delta ve eğitilmiş model gereklidir):

```bash
docker compose run --rm --no-deps spark /opt/spark/bin/spark-submit --driver-memory 4g --conf spark.driver.maxResultSize=2g --packages io.delta:delta-spark_2.12:3.2.0 /app/jobs/export_best_predictions.py
```

Üretilen çıktı:

```text
reports/ml_outputs/best_model_predictions.csv   ← 40.000 satır, probability skorları
```

Dashboard'u başlatmak için:

```bash
docker compose up -d --build dashboard
```

Tarayıcıdan erişim: `http://localhost:8501`

Dashboard sayfaları ve içerikleri:

| Sayfa | İçerik |
|---|---|
| Landing (app.py) | Pipeline mimarisi, 4 özet metrik kartı |
| 01 EDA | Sentiment dağılımı (pie), histogram, area chart, stacked bar, scatter, 10 bölüm |
| 02 Features | TF-IDF pipeline diyagramı, Top-20 feature grafikleri (RF/GBT/DT) |
| 03 Model Comparison | 5×5 grouped bar chart, metrik tablosu, eğitim koşulları |
| 04 Best Model | Confusion matrix (renkli), ROC curve (AUC=0.9424), feature importance |

---

## Önemli Notlar

- `data/raw/steam_reviews.csv` büyük veri dosyası olduğu için GitHub'a gönderilmez.
- `data/delta/` ve `data/checkpoints/` çalışma sırasında oluşan büyük veri çıktılarıdır ve GitHub'a gönderilmez.
- Projeyi farklı bilgisayarda çalıştırmak için `data/raw/steam_reviews.csv` dosyasının manuel olarak `data/raw/` klasörüne eklenmesi gerekir.
- Docker image indirme hataları proje kodundan değil, Docker Hub / internet / DNS bağlantısından kaynaklanabilir.