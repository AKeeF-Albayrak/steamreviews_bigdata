import pandas as pd
import plotly.express as px
import streamlit as st

from lib.io import read_ml_csv
from lib.viz import NEGATIVE_COLOR, POSITIVE_COLOR, model_comparison_bar

st.set_page_config(page_title="Model Karsilastirma", layout="wide")
st.title("Coklu Model Karsilastirmasi")
st.markdown("*5 farkli siniflandirma modeli MLflow ile takip edildi; metrikler AUC-ROC onceliklendirilerek karsilastirildi.*")
st.markdown("5 farkli siniflandirma modeli egitildi ve MLflow ile takip edildi.")
st.markdown("---")

comparison_df = read_ml_csv("model_comparison.csv")

if comparison_df is None:
    st.error("model_comparison.csv bulunamadi.")
    st.stop()

# --- 5 model ozet kartlari ---
st.subheader("Model Performans Ozeti")
st.markdown("Her modelin AUC-ROC ve Accuracy degerleri. En yuksek AUC'ye sahip model **BEST** olarak isaretlenmistir.")

short_names = {
    "Logistic Regression Existing": "LR",
    "Decision Tree Classifier": "DT",
    "Random Forest Classifier": "RF",
    "Gradient Boosted Trees Classifier": "GBT",
    "Naive Bayes": "NB",
}

sorted_df = comparison_df.sort_values("area_under_roc", ascending=False)

# 3+2 grid layout (dar ekran dostu)
row1_cols = st.columns(3)
row2_cols = st.columns(2)
all_cols = row1_cols + row2_cols

for col, (_, row) in zip(all_cols, sorted_df.iterrows()):
    short = short_names.get(row["model_name"], row["model_name"])
    is_best = row["model_name"] == sorted_df.iloc[0]["model_name"]
    label = f"{short} {'★ BEST' if is_best else ''}"
    col.metric(
        label=label,
        value=f"AUC {row['area_under_roc']:.4f}",
        delta=f"Acc {row['accuracy']:.4f}",
    )

st.markdown("---")

# --- Grouped bar chart ---
st.subheader("5 Model x 5 Metrik Karsilastirmasi")
st.markdown(
    "Metrikler onem sirasiyla soldan saga dizilmistir: AUC-ROC → F1-Score → Accuracy → Recall → Precision. "
    "Her bar uzerindeki deger dogrudan okunabilir."
)
st.plotly_chart(model_comparison_bar(comparison_df), use_container_width=True)

st.markdown("---")

# --- Detayli metrik tablosu ---
st.subheader("Detayli Metrik Tablosu")
st.markdown("AUC-ROC'a gore sirali tam metrik tablosu. Sutun basliklarinda siralama yapilabilir.")
display_df = comparison_df.copy()
display_df.columns = ["Model", "Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC"]
display_df = display_df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
display_df.index = display_df.index + 1

st.dataframe(
    display_df.style
    .highlight_max(subset=["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC"], color="#d4edda")
    .format({"Accuracy": "{:.4f}", "F1-Score": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "AUC-ROC": "{:.4f}"}),
    use_container_width=True,
)

st.markdown("---")

# --- Model egitim kosullari ---
st.subheader("Model Egitim Kosullari")
st.markdown("Modeller farkli kosullarda egitildi. Bu farkliliklari anlamak, sonuclari dogru yorumlamak icin kritiktir.")
st.warning(
    "Logistic Regression Existing, diger 4 modelden farkli kosullarda egitildi: "
    "100k ornek/sinif ve HashingTF(20000) pipeline'i kullanildi. "
    "Diger modeller 25k ornek/sinif ve CountVectorizer(5000) pipeline'i kullanildi. "
    "Bu nedenle LR'nin yuksek AUC degeri kismen egitim avantajindan kaynaklanmaktadir."
)

model_props = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "GBT", "Naive Bayes"],
    "Egitim Ornegi/Sinif": ["100.000", "25.000", "25.000", "25.000", "25.000"],
    "Feature Pipeline": ["HashingTF(20000)", "CountVectorizer(5000)", "CountVectorizer(5000)",
                         "CountVectorizer(5000)", "CountVectorizer(5000)"],
    "Hiperparametreler": [
        "maxIter=20, regParam=0.01",
        "maxDepth=5",
        "numTrees=20, maxDepth=5",
        "maxIter=20, maxDepth=5",
        "smoothing=1.0, multinomial",
    ],
    "Hiperparametre Tuning": ["Hayir"] * 5,
})
st.dataframe(model_props, use_container_width=True, hide_index=True)

st.markdown("---")

# --- Sinif tahmin dagilimi ---
st.subheader("Tahmin Dagilimi (LR Existing — Test Seti)")
st.markdown("Modelin test setinde kac yorum pozitif, kac yorum negatif olarak tahmin ettigini gosterir.")
pred_dist = read_ml_csv("prediction_distribution.csv")
if pred_dist is not None:
    fig = px.bar(
        pred_dist,
        x="prediction_text",
        y="count",
        color="prediction_text",
        color_discrete_map={"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR},
        title="Test Setinde Tahmin Dagilimi",
        labels={"prediction_text": "Tahmin", "count": "Sayi"},
        text_auto=",",
    )
    fig.update_layout(showlegend=False)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        total = pred_dist["count"].sum()
        st.markdown("#### Yorum")
        st.markdown(
            f"Test setinde toplam **{total:,}** tahmin yapildi. "
            "Dengeli veri ile egitildiginden tahminler iki sinif arasinda yaklasik esit dagildi."
        )
