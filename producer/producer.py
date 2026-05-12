import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "steam_reviews")
DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/steam_reviews.csv"))
MESSAGE_DELAY_SECONDS = float(os.getenv("MESSAGE_DELAY_SECONDS", "0.1"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "1000"))


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda key: str(key).encode("utf-8"),
        retries=5,
    )


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_event(row, event_id: int) -> dict:
    review_score = int(row["review_score"])

    if review_score == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "event_id": event_id,
        "event_type": "steam_review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app_id": int(row["app_id"]),
        "app_name": clean_text(row["app_name"]),
        "review_text": clean_text(row["review_text"]),
        "review_score": review_score,
        "review_votes": int(row["review_votes"]) if not pd.isna(row["review_votes"]) else 0,
        "sentiment": sentiment,
    }


def main():
    print("Steam Reviews Producer başlatılıyor...")
    print(f"Kafka server: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Mesaj aralığı: {MESSAGE_DELAY_SECONDS} saniye")
    print(f"Maksimum mesaj: {MAX_MESSAGES}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATA_PATH}")

    producer = create_producer()

    sent_count = 0

    for chunk in pd.read_csv(DATA_PATH, chunksize=1000):
        for _, row in chunk.iterrows():
            if sent_count >= MAX_MESSAGES:
                producer.flush()
                print(f"Producer tamamlandı. Toplam gönderilen mesaj: {sent_count}")
                return

            event = build_event(row, sent_count + 1)

            producer.send(
                topic=KAFKA_TOPIC,
                key=event["app_id"],
                value=event,
            )

            sent_count += 1

            if sent_count % 100 == 0:
                print(f"{sent_count} mesaj gönderildi...")

            time.sleep(MESSAGE_DELAY_SECONDS)

    producer.flush()
    print(f"Dataset bitti. Toplam gönderilen mesaj: {sent_count}")


if __name__ == "__main__":
    main()