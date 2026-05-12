FROM apache/spark:3.5.1-python3

WORKDIR /app

COPY spark/jobs /app/jobs

USER root
RUN mkdir -p /app/delta /app/checkpoints /app/reports/figures /app/reports/eda_outputs
RUN pip install pandas matplotlib