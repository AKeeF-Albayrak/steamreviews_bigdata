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