FROM python:3.11-slim

WORKDIR /app

COPY producer/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY producer/producer.py /app/producer.py

CMD ["python", "producer.py"]