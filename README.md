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

---

## 1. Proje Klasör Yapısı

Proje ana dizini:

```text
steamreviews_bigdata/
├── data/
│   ├── raw/
│   ├── sample/
│   └── processed/
├── producer/
│   ├── producer.py
│   ├── Dockerfile
│   └── requirements.txt
├── spark/
│   ├── jobs/
│   └── utils/
├── notebooks/
│   └── check_dataset.py
├── ml/
├── dashboard/
├── reports/
├── docs/
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore

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

Streaming job sürekli çalışan bir yapıdadır. İşlem doğrulandıktan sonra durdurmak için:

```bash
docker stop steam_spark_streaming
```

---

## Adım 4 - Keşifsel Veri Analizi (EDA)

Bu adımda Silver Delta katmanındaki temizlenmiş veri okunarak keşifsel veri analizi yapılmıştır.

EDA dosyası:

```text
spark/jobs/eda_steam_reviews.py
```

Çalıştırma komutu:

```bash
docker compose run --rm spark /opt/spark/bin/spark-submit --packages io.delta:delta-spark_2.12:3.2.0 /app/jobs/eda_steam_reviews.py
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
