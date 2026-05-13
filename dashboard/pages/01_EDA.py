import pandas as pd
import plotly.express as px
import streamlit as st

from lib.io import read_eda_csv, read_ml_csv, read_silver_sample
from lib.viz import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    app_sentiment_stacked_bar,
    sentiment_donut,
    text_length_bar,
    text_votes_scatter,
    top_apps_bar,
)

st.set_page_config(page_title="EDA", layout="wide")
st.title("Kesifsel Veri Analizi (EDA)")
st.markdown("*Silver Delta katmanindaki 5.000.000 Steam yorumu uzerinde yapilan analiz.*")
st.markdown("---")

# --- Ozet metrikler ---
summary = read_eda_csv("basic_summary.csv")
missing = read_eda_csv("missing_values.csv")

if summary is not None:
    s = summary.set_index("metric")["value"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Yorum", f"{int(s['total_rows']):,}")
    c2.metric("Benzersiz Oyun", f"{int(s['unique_apps']):,}")
    c3.metric("Sinif Sayisi", int(s["unique_sentiments"]))
    missing_total = int(missing.iloc[0].sum()) if missing is not None else 0
    c4.metric(
        "Eksik Deger",
        missing_total,
        delta="Silver temizligi basarili" if missing_total == 0 else None,
    )

st.markdown("---")

# --- 1. Sentiment dagilimi ---
st.subheader("1. Sentiment Dagilimi")
st.markdown("Veri setindeki pozitif ve negatif yorum oranlarini gosterir. Sinif dengesizligi ML egitimini dogrudan etkiler.")
sentiment_df = read_eda_csv("sentiment_distribution.csv")
if sentiment_df is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(sentiment_donut(sentiment_df), use_container_width=True)
    with col2:
        st.markdown("#### Sinif Dengesizligi")
        total = sentiment_df["count"].sum()
        for _, row in sentiment_df.iterrows():
            pct = row["count"] / total * 100
            color = POSITIVE_COLOR if row["sentiment"] == "positive" else NEGATIVE_COLOR
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:8px;margin:6px 0;"
                f"color:white;font-size:16px'>"
                f"<b>{row['sentiment'].capitalize()}</b>: {int(row['count']):,} yorum ({pct:.1f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.info(
            "Veri seti %81 positive / %19 negative oraniyla dengesizdir. "
            "ML egitiminde her siniftan 25.000 ornek alinarak dengelendi."
        )

st.markdown("---")

# --- 2. Review score dagilimi ---
st.subheader("2. Review Score Dagilimi")
st.markdown("Steam'in ham puan sistemi: +1 (positive) ve -1 (negative). Sentiment etiketi bu alandan turetilmistir.")
score_df = read_eda_csv("review_score_distribution.csv")
if score_df is not None:
    score_df["score_label"] = score_df["review_score"].map({1: "Positive (1)", -1: "Negative (-1)"})
    fig = px.bar(
        score_df,
        x="score_label",
        y="count",
        color="score_label",
        color_discrete_map={"Positive (1)": POSITIVE_COLOR, "Negative (-1)": NEGATIVE_COLOR},
        title="Review Score Dagilimi (-1 / +1)",
        labels={"score_label": "Skor", "count": "Yorum Sayisi"},
        text_auto=",",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 3. Top oyunlar ---
st.subheader("3. En Cok Yorum Alan Oyunlar")
st.markdown("Platformdaki aktif topluluk boyutunu gosteren metrik. PAYDAY 2 ve DayZ on siradadir.")
top_apps = read_eda_csv("top_apps.csv")
if top_apps is not None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(top_apps_bar(top_apps, top_n=15), use_container_width=True)
    with col2:
        st.markdown("#### Top 5 — Ortalama Helpful Votes")
        top5 = top_apps.nlargest(5, "avg_review_votes")[["app_name", "avg_review_votes"]]
        top5["avg_review_votes"] = top5["avg_review_votes"].round(3)
        top5.columns = ["Oyun", "Ort. Oy"]
        st.dataframe(top5, use_container_width=True, hide_index=True)
        st.caption("Grand Theft Auto V en yuksek ortalama helpful vote'a sahip (0.225).")

st.markdown("---")

# --- 4. Top 10 oyun sentiment dagilimi ---
st.subheader("4. Oyun Bazinda Sentiment Orani")
st.markdown("Her oyunun kendi yorum havuzunda pozitif/negatif oraninin nasil degistigini gosterir.")
silver_df = read_silver_sample()
if silver_df is not None:
    st.plotly_chart(app_sentiment_stacked_bar(silver_df, top_n=10), use_container_width=True)
    st.caption("Silver Delta ornegininden hesaplanmistir. Oyun baziyla oranlar toplam veri setinden hafifce sapabilir.")
else:
    st.info("Silver Delta verisi bulunamadi. Docker volume baglantilarini kontrol edin.")

st.markdown("---")

# --- 5. Yorum uzunlugu ---
st.subheader("5. Yorum Uzunlugu: Positive vs Negative")
st.markdown("Negatif yorumlarin ortalama olarak daha uzun oldugu gozlemlenmistir — mutsuz kullanicilar daha ayrintili yazma egilimindedir.")
length_df = read_eda_csv("review_text_length_summary.csv")
if length_df is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(text_length_bar(length_df), use_container_width=True)
    with col2:
        st.markdown("#### Bulgu")
        neg_len = length_df[length_df["sentiment"] == "negative"]["avg_review_text_length"].values[0]
        pos_len = length_df[length_df["sentiment"] == "positive"]["avg_review_text_length"].values[0]
        diff = neg_len - pos_len
        st.metric("Negative ort. uzunluk", f"{neg_len:.0f} karakter")
        st.metric("Positive ort. uzunluk", f"{pos_len:.0f} karakter")
        st.metric("Fark", f"+{diff:.0f} karakter (negative daha uzun)")
        st.info(
            "Negatif yorumlar ortalama 97 karakter daha uzun. "
            "Mutsuz kullanicilar daha ayrintili yorum yazma egiliminde."
        )

st.markdown("---")

# --- 6. Yorum uzunlugu histogram ---
st.subheader("6. Yorum Uzunlugu Dagilimi (Histogram)")
st.markdown("Bireysel yorum uzunluklarinin dagilimi: negatif yorumlarin kuyrugunun daha uzun oldugunu gosterir.")
pred_df = read_ml_csv("best_model_predictions.csv")
if pred_df is not None and "review_text" in pred_df.columns:
    pred_df["text_len"] = pred_df["review_text"].str.len()

    col1, col2 = st.columns([3, 1])
    with col1:
        fig_hist = px.histogram(
            pred_df,
            x="text_len",
            color="label_text",
            nbins=60,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR},
            title="Yorum Karakteri Uzunlugu Dagilimi (40k ornek)",
            labels={"text_len": "Karakter Sayisi", "label_text": "Sentiment", "count": "Yorum Sayisi"},
        )
        # P25 / P50 / P75 dikey cizgiler
        for q, lbl, clr in [(0.25, "P25", "gray"), (0.50, "P50 (Medyan)", "#2c3e50"), (0.75, "P75", "gray")]:
            qv = pred_df["text_len"].quantile(q)
            fig_hist.add_vline(x=qv, line_dash="dot", line_color=clr, line_width=1.5,
                               annotation_text=f"{lbl}={qv:.0f}", annotation_position="top right",
                               annotation_font_size=9)
        fig_hist.update_layout(legend_title="Sentiment")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("#### Dagilim Istatistikleri")
        for sentiment, clr in [("positive", POSITIVE_COLOR), ("negative", NEGATIVE_COLOR)]:
            grp = pred_df[pred_df["label_text"] == sentiment]["text_len"]
            st.markdown(
                f"<div style='background:{clr};padding:8px;border-radius:6px;margin:4px 0;color:white;font-size:13px'>"
                f"<b>{sentiment.capitalize()}</b><br>"
                f"Ort: {grp.mean():.0f} | Std: {grp.std():.0f}<br>"
                f"P50: {grp.median():.0f} | P75: {grp.quantile(0.75):.0f}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption("Negatif yorumlar daha sag-kayik (uzun kuyruk) bir dagilim sergiler.")

st.markdown("---")

# --- 7. Yorum uzunlugu vs helpful votes scatter ---
st.subheader("7. Yorum Uzunlugu vs Helpful Votes")
st.markdown("Uzun yorumlar daha fazla 'helpful' oyu aliyor mu? Trend cizgisi genel iliskiyi gosterir.")
if silver_df is not None:
    st.plotly_chart(text_votes_scatter(silver_df, n_sample=3000), use_container_width=True)
    st.caption(
        "Silver Delta orneginden 3.000 rastgele kayit. "
        "Helpful votes 0 olan yorumlar dahil edilmistir; log-scale y-ekseni yoktur, "
        "bu nedenle yuksek oylu istisnai yorumlar grafigi domine edebilir."
    )
else:
    st.info("Silver verisi bulunamadi.")

st.markdown("---")

# --- 8. Eksik deger analizi ---
st.subheader("8. Eksik Deger Analizi")
st.markdown("Silver katmaninda uygulanan null filtresinin etkinligini dogrular.")
if missing is not None:
    st.success("Silver katmaninda hicbir kolonda eksik deger bulunmamaktadir.")
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.caption("Spark streaming job null filtresi uyguladigi icin tum eksik degerler temizlendi.")

st.markdown("---")

# --- 9. Veri akis trendi ---
st.subheader("9. Veri Akis Trendi (Ingestion Zamani)")
st.markdown("Kafka→Spark pipeline'inin kac yorumu saat basina isledigini gosterir.")
st.warning(
    "Bu grafik review submission zamanini DEGIL, ingestion zamanini gostermektedir. "
    "Producer datetime.now() kullandigi icin tum yorumlar 2 saatlik pencereye sikisti."
)
hourly = read_eda_csv("hourly_trend.csv")
if hourly is not None:
    fig_area = px.area(
        hourly,
        x="hour",
        y="count",
        markers=True,
        title="Saatlik Ingestion Trendi (Pipeline Performans Kaniti)",
        labels={"hour": "Saat (Ingestion)", "count": "Islenen Yorum Sayisi"},
    )
    fig_area.update_traces(
        marker=dict(size=8),
        line=dict(width=2),
        fillcolor="rgba(31, 119, 180, 0.2)",
    )
    st.plotly_chart(fig_area, use_container_width=True)

st.markdown("---")

# --- 10. Sinif dagilimi gercek vs ML ---
st.subheader("10. Gercek Sinif Dagilimi vs ML Egitim Verisi")
st.markdown(
    "5M satirlik ham verideki dengesizlik ile ML icin olusturulan dengeli egitim setinin karsilastirilmasi. "
    "Undersample yontemi: her siniftan limit(25.000) alinmistir."
)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Gercek veri (5M satir)**")
    df_real = pd.DataFrame({"Sinif": ["Positive", "Negative"], "Sayi": [4_067_557, 932_443]})
    fig_real = px.bar(
        df_real, x="Sinif", y="Sayi",
        color="Sinif",
        color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
        text_auto=",",
    )
    fig_real.update_layout(showlegend=False)
    st.plotly_chart(fig_real, use_container_width=True)

with col2:
    st.markdown("**ML egitim verisi (dengeli, 25k/sinif)**")
    df_bal = pd.DataFrame({"Sinif": ["Positive", "Negative"], "Sayi": [25_000, 25_000]})
    fig_bal = px.bar(
        df_bal, x="Sinif", y="Sayi",
        color="Sinif",
        color_discrete_map={"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR},
        text_auto=",",
    )
    fig_bal.update_layout(showlegend=False)
    st.plotly_chart(fig_bal, use_container_width=True)

st.caption(
    "Sinif dengesizligini gidermek icin her siniftan 25.000 ornek alindi (limit-based undersampling). "
    "Alternatif: weightCol ile agirlikli kayip fonksiyonu."
)
