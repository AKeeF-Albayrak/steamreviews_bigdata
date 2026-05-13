import pandas as pd
import streamlit as st

from lib.io import ml_file_exists, read_best_model_summary, read_ml_csv
from lib.viz import confusion_matrix_heatmap, feature_importance_bar, roc_curve_chart

st.set_page_config(page_title="En Iyi Model", layout="wide")
st.title("En Iyi Model: Logistic Regression")
st.markdown("*Logistic Regression (LR Existing) en yuksek AUC-ROC (0.9424) ve Accuracy (0.8794) elde etti.*")
st.markdown("---")

# --- 1. Model kunyesi ---
best = read_best_model_summary()

st.subheader("1. Model Kunyesi")
st.markdown("Egitim konfigurasyonu, pipeline parametreleri ve elde edilen metrikler.")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{float(best.get('accuracy', 0)):.4f}")
col2.metric("F1-Score", f"{float(best.get('f1_score', 0)):.4f}")
col3.metric("Precision", f"{float(best.get('weighted_precision', 0)):.4f}")
col4.metric("Recall", f"{float(best.get('weighted_recall', 0)):.4f}")
col5.metric("AUC-ROC", f"{float(best.get('area_under_roc', 0)):.4f}")

model_info = pd.DataFrame({
    "Parametre": [
        "Script", "Feature Pipeline", "Egitim Ornegi/Sinif",
        "maxIter", "regParam", "Stratified Split", "Hiperparametre Tuning"
    ],
    "Deger": [
        "spark/jobs/train_sentiment_model.py",
        "Tokenizer → StopWordsRemover → HashingTF(20000) → IDF",
        "100.000",
        "20", "0.01", "Hayir (randomSplit seed=42)", "Hayir"
    ],
})
st.dataframe(model_info, use_container_width=True, hide_index=True)

st.markdown("---")

# --- 2. Confusion Matrix ---
st.subheader("2. Confusion Matrix")
st.markdown(
    "Her hucre farkli renkte: **yesil=TN**, **mavi=TP**, **turuncu=FP**, **kirmizi=FN**. "
    "FP ve FN oranlarinin benzer olmasi modelin dengeli hata yaptigini gosterir."
)
cm_df = read_ml_csv("confusion_matrix_logistic_regression_existing.csv")

if cm_df is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            confusion_matrix_heatmap(cm_df, "Logistic Regression Existing"),
            use_container_width=True,
        )
    with col2:
        pivot = cm_df.pivot(index="label_text", columns="prediction_text", values="count").fillna(0)
        tn = int(pivot.loc["negative", "negative"])
        fp = int(pivot.loc["negative", "positive"])
        fn = int(pivot.loc["positive", "negative"])
        tp = int(pivot.loc["positive", "positive"])
        total = tn + fp + fn + tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        st.markdown("#### Hesaplanan Metrikler")
        metrics_df = pd.DataFrame({
            "Metrik": [
                "True Positive (TP)", "True Negative (TN)",
                "False Positive (FP)", "False Negative (FN)",
                "Precision", "Recall (Sensitivity)",
                "Specificity", "Accuracy", "F1-Score",
            ],
            "Deger": [
                f"{tp:,}", f"{tn:,}", f"{fp:,}", f"{fn:,}",
                f"{precision:.4f}", f"{sensitivity:.4f}",
                f"{specificity:.4f}", f"{accuracy:.4f}", f"{f1:.4f}",
            ],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Model negatif yorumlarin %{fp/(tn+fp)*100:.1f}'ini yanlis pozitif tahmin ediyor (FP={fp:,}). "
            f"Pozitif yorumlarin %{fn/(tp+fn)*100:.1f}'ini kaciriyor (FN={fn:,}). "
            "Hata oranlari birbirine yakin: model dengeli hata yapiyor."
        )

st.markdown("---")

# --- 3. ROC Curve ---
st.subheader("3. ROC Curve")
st.markdown(
    "Sol panel: AUC altindaki alan golgelenmis, optimal threshold yesil yildiz ile isaretlenmis, "
    "0.3/0.5/0.7 esikleri turuncu noktalar ile gosterilmistir. "
    "Sag panel: pozitif ve negatif siniflar icin model olasilik skorlari dagilimi."
)

if ml_file_exists("best_model_predictions.csv"):
    pred_df = read_ml_csv("best_model_predictions.csv")
    if pred_df is not None:
        st.plotly_chart(roc_curve_chart(pred_df), use_container_width=True)
        st.caption(
            "Sag histogramda iki sinifin skorlari ne kadar ayrisiyorsa model o kadar iyi ayirt ediyor. "
            "Optimal threshold Youden Index (max TPR-FPR) ile belirlenmistir."
        )
else:
    st.info(
        "ROC Curve icin best_model_predictions.csv henuz olusturulmadi. "
        "Asagidaki komutu calistirarak uret:\n\n"
        "```\n"
        "docker compose run --rm --no-deps spark /opt/spark/bin/spark-submit "
        "--driver-memory 4g --conf spark.driver.maxResultSize=2g "
        "--packages io.delta:delta-spark_2.12:3.2.0 "
        "/app/jobs/export_best_predictions.py\n"
        "```"
    )
    auc_val = float(best.get("area_under_roc", 0.9424))
    st.metric("AUC-ROC (MLflow'dan)", f"{auc_val:.4f}")

st.markdown("---")

# --- 4. Feature Importance ---
st.subheader("4. Feature Importance")
st.markdown(
    "LR Existing modeli HashingTF kullandigi icin feature isimleri **hash_feature_XXXX** formatindadir; "
    "hangi kelimeye karsilik geldigi bilinmez. "
    "Gercek kelime gormek icin `02_Features` sayfasindaki RF/GBT grafikleri incelenebilir."
)
fi_df = read_ml_csv("feature_importance_logistic_regression_existing.csv")

if fi_df is not None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            feature_importance_bar(fi_df, top_n=20, title="Top 20 Feature — LR Existing (HashingTF)"),
            use_container_width=True,
        )
    with col2:
        st.warning(
            "LR Existing modeli **HashingTF** kullandigi icin feature isimleri "
            "`hash_feature_XXXX` formatindadir ve hangi kelimeye karsili geldigi bilinmez. "
            "Diger modeller (RF, GBT) CountVectorizer kullandigi icin gercek kelimeler gosterilir."
        )
        st.markdown("#### Pozitif / Negatif Katkili Ozellikler")
        pos_fi = fi_df[fi_df["importance"] > 0].nlargest(5, "importance")
        neg_fi = fi_df[fi_df["importance"] < 0].nsmallest(5, "importance")
        st.markdown("**Positive sinifa katkili (yesil):**")
        for _, r in pos_fi.iterrows():
            st.markdown(f"- `{r['feature']}`: {r['importance']:.3f}")
        st.markdown("**Negative sinifa katkili (kirmizi):**")
        for _, r in neg_fi.iterrows():
            st.markdown(f"- `{r['feature']}`: {r['importance']:.3f}")

st.markdown("---")

# --- 5. Ornek tahminler ---
st.subheader("5. Ornek Tahminler")
st.markdown("Modelin dogru ve yanlis siniflandirdigi gercek yorumlar. Yanlis tahminler neden yanlis olabilir?")

if ml_file_exists("best_model_predictions.csv"):
    pred_df = read_ml_csv("best_model_predictions.csv")
    if pred_df is not None and "review_text" in pred_df.columns:
        correct = pred_df[pred_df["label_text"] == pred_df["prediction_text"]].head(5)
        wrong = pred_df[pred_df["label_text"] != pred_df["prediction_text"]].head(5)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Dogru Tahminler")
            for _, row in correct.iterrows():
                color = "#2ecc71" if row["label_text"] == "positive" else "#e74c3c"
                text_preview = str(row["review_text"])[:200] + "..."
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:8px;margin:6px 0;"
                    f"background:#f9f9f9;border-radius:4px;font-size:13px'>"
                    f"<b>{row['label_text'].upper()}</b> → tahmin: {row['prediction_text']}<br>"
                    f"<i>{text_preview}</i></div>",
                    unsafe_allow_html=True,
                )
        with col2:
            st.markdown("#### Yanlis Tahminler")
            for _, row in wrong.iterrows():
                text_preview = str(row["review_text"])[:200] + "..."
                st.markdown(
                    f"<div style='border-left:4px solid #f39c12;padding:8px;margin:6px 0;"
                    f"background:#fff9f0;border-radius:4px;font-size:13px'>"
                    f"<b>Gercek: {row['label_text'].upper()}</b> → tahmin: {row['prediction_text']}<br>"
                    f"<i>{text_preview}</i></div>",
                    unsafe_allow_html=True,
                )
else:
    st.info("Ornek tahminler icin once export_best_predictions.py calistirilmalidir.")
