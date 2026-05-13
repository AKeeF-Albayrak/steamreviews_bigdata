import streamlit as st

from lib.io import read_eda_csv, read_best_model_summary

st.set_page_config(
    page_title="Steam Yorumlari - Buyuk Veri Analizi",
    layout="wide",
)

st.title("Steam Yorumlari Buyuk Veri Analizi")
st.markdown("Kafka + Spark Structured Streaming + Delta Lake + PySpark MLlib + MLflow")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Sistem Mimarisi")
    st.code(
        """
steam_reviews.csv (2 GB, 5M yorum)
          |
          v
  Python Kafka Producer
          |
          v
  Apache Kafka  (steam_reviews topic)
          |
          v
  Spark Structured Streaming
          |
    +-----+-----+
    |           |
    v           v
Bronze Delta  (ham JSON)
    |
    v
Silver Delta  (temizlenmis, 5M satir)
    |
    +----------+----------+
    |                     |
    v                     v
EDA (reports/)      ML Egitimi (5 Model)
                          |
                    MLflow File-Store
                    (reports/mlruns/)
                          |
                          v
                    Streamlit Dashboard  
""",
        language=None,
    )

with col2:
    st.subheader("Proje Metrikleri")

    summary = read_eda_csv("basic_summary.csv")
    best = read_best_model_summary()

    if summary is not None:
        s = summary.set_index("metric")["value"]
        m1, m2 = st.columns(2)
        m1.metric("Toplam Yorum", f"{int(s['total_rows']):,}")
        m2.metric("Benzersiz Oyun", f"{int(s['unique_apps']):,}")

    m3, m4 = st.columns(2)
    model_name = best.get("model_name", "Logistic Regression Existing")
    auc_val = best.get("area_under_roc", "0.9424")
    m3.metric("En Iyi Model", model_name.replace("Existing", "").strip())
    m4.metric("En Iyi AUC-ROC", f"{float(auc_val):.4f}")

    st.markdown("---")
    st.subheader("Veri Seti")
    st.markdown(
        """
| Ozellik | Deger |
|---|---|
| Kaynak | Kaggle — Steam Reviews |
| Boyut | ~5.000.000 satir |
| Hedef Degisken | `sentiment` (positive / negative) |
| Problem Turu | Binary Classification |
| Sinif Dagilimi | %81 positive / %19 negative |
| Benzersiz Oyun | 5.580 |
"""
    )

st.markdown("---")
st.info("Sol menuden bir sayfa secin: EDA, Ozellik Muhendisligi, Model Karsilastirma, En Iyi Model.")
