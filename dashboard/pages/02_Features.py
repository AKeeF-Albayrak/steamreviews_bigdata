import pandas as pd
import streamlit as st

from lib.io import read_ml_csv
from lib.viz import NEGATIVE_COLOR, POSITIVE_COLOR, feature_importance_bar

st.set_page_config(page_title="Ozellik Muhendisligi", layout="wide")
st.title("Ozellik Muhendisligi (Feature Engineering)")
st.markdown("*TF-IDF tabanli metin isleme pipeline'i ile kelime bazli ozellik uretimi.*")
st.markdown("---")

# --- 1. Text Feature Pipeline ---
st.subheader("1. Text Feature Pipeline")
st.markdown(
    "Tum modeller icin ortak TF-IDF tabanli metin onisleme pipeline'i kullanildi "
    "(`spark/jobs/train_compare_models_mlflow.py`, satirlar 71-110). "
    "Her asama ham metni model icin hazir sayisal vektore donusturur."
)

stages = [
    ("StringIndexer", "label_text → label", "alphabetAsc: negative=0, positive=1"),
    ("Tokenizer", "review_text → words", "Bosluk bazli tokenizasyon"),
    ("StopWordsRemover", "words → filtered_words", "Ingilizce stop-word listesi uygulanir"),
    ("CountVectorizer", "filtered_words → tf_features", "vocabSize=5000, minDF=5 (en az 5 dokumanda gecen kelimeler)"),
    ("IDF", "tf_features → features", "Nadir kelimelere yuksek agirlik, yaygin kelimelere dusuk agirlik"),
    ("Classifier", "features → prediction", "Model tipine gore degisir (LR / DT / RF / GBT / NB)"),
]

df_stages = pd.DataFrame(stages, columns=["Asama", "Donusum", "Parametre / Aciklama"])
st.dataframe(df_stages, use_container_width=True, hide_index=True)

st.markdown("---")

# --- 2. Pipeline akis semasi ---
st.subheader("2. Pipeline Akis Semasi")
st.markdown("Her yorum asagidaki adimlardan gecirilerek 5000 boyutlu TF-IDF vektorune donusturulur.")
st.code(
    """
review_text (ham metin)
      |
      v
  Tokenizer
  "great game fun" → ["great", "game", "fun"]
      |
      v
  StopWordsRemover
  ["great", "game", "fun"] → ["great", "game", "fun"]  (stop words cikarilir)
      |
      v
  CountVectorizer  (vocabSize=5000, minDF=5)
  Kelime frekans vektoru → sparse TF vektoru (5000 boyutlu)
      |
      v
  IDF (Inverse Document Frequency)
  TF vektoru x IDF agirliklari → TF-IDF features
      |
      v
  Classifier (LR / DT / RF / GBT / NB)
  features → prediction (0=negative / 1=positive)
""",
    language=None,
)

st.markdown("---")

# --- 3. Top features ---
st.subheader("3. En Onemli Ozellikler (Top Vocabulary)")
st.markdown(
    "Tree-based modeller (RF, GBT, DT) CountVectorizer kullandigi icin feature isimleri **gercek Ingilizce kelimelerdir**. "
    "LR Existing ise HashingTF kullandigi icin feature isimleri hash kodu formatindadir (okunaksiz)."
)

tabs = st.tabs(["Random Forest", "Gradient Boosted Trees", "Decision Tree"])

model_files = {
    "Random Forest": "feature_importance_random_forest_classifier.csv",
    "Gradient Boosted Trees": "feature_importance_gradient_boosted_trees_classifier.csv",
    "Decision Tree": "feature_importance_decision_tree_classifier.csv",
}

for tab, (model_label, fname) in zip(tabs, model_files.items()):
    with tab:
        fi_df = read_ml_csv(fname)
        if fi_df is not None:
            st.plotly_chart(
                feature_importance_bar(
                    fi_df,
                    top_n=20,
                    title=f"Top 20 En Etkili Kelime — {model_label} (Sentiment Tahmininde En Yuksek Agirlikli)",
                ),
                use_container_width=True,
            )
            st.caption(
                f"{model_label}: CountVectorizer(5000) kullanildi. "
                "Yesil barlar pozitif sinifa katkili, kirmizi barlar negatif sinifa katkili ozellikleri gosterir."
            )
        else:
            st.info(f"{fname} henuz mevcut degil.")

st.markdown("---")

# --- 4. Modeller arasi top-5 feature ---
st.subheader("4. Modeller Arasi Top-5 Feature Karsilastirmasi")
st.markdown(
    "Her modelin en onemli 5 ozelligini yan yana listeler. "
    "LR Existing (HashingTF) hash kodlari gosterirken, diger modeller gercek kelimeler gosterir."
)
st.caption(
    "LR Existing modeli HashingTF kullandigi icin feature isimleri hash_feature_XXXX formatindadir "
    "(okunamaz). Diger modeller CountVectorizer kullandigi icin gercek kelimeler gosterilir."
)

all_fi = read_ml_csv("feature_importance_all_models.csv")
if all_fi is not None:
    comparison_rows = []
    for model_name in all_fi["model_name"].unique():
        model_df = all_fi[all_fi["model_name"] == model_name].copy()
        model_df["abs_importance"] = model_df["importance"].abs()
        top5 = model_df.nlargest(5, "abs_importance")["feature"].tolist()
        short = (
            model_name.replace("Classifier", "").replace("Existing", "").replace("  ", " ").strip()
        )
        comparison_rows.append({"Model": short, "Top-5 Feature": ", ".join(top5)})

    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# --- 5. Sinirlamalar ---
st.subheader("5. Sinirlamalar ve Gelecek Calisma")
st.warning(
    "Bu projede yalnizca metin tabanli TF-IDF ozellikleri kullanildi. "
    "Asagidaki sayisal/zamansal ozellikler eklenerek model performansi artirilabildigi dusunulmektedir:"
)

future_features = pd.DataFrame({
    "Onerilen Feature": [
        "review_text_length",
        "review_text_word_count",
        "review_votes",
        "app_review_count",
        "app_positive_ratio",
    ],
    "Aciklama": [
        "Yorumun karakter sayisi (negative yorumlar daha uzun) — EDA'da kanitlandi",
        "Yorum kelime sayisi",
        "Diger kullanicilarin helpful oyu",
        "Oyunun toplam yorum sayisi (populerlik gostergesi)",
        "Oyun bazinda positive yorum orani (dikkat: data leakage riski)",
    ],
    "Tip": ["Sayisal", "Sayisal", "Sayisal", "Sayisal", "Sayisal"],
})
st.dataframe(future_features, use_container_width=True, hide_index=True)
