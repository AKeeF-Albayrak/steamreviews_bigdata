import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc

POSITIVE_COLOR = "#2ecc71"
NEGATIVE_COLOR = "#e74c3c"
MODEL_COLORS = px.colors.qualitative.Dark2

DEFAULT_CHART_HEIGHT = 450
SMALL_CHART_HEIGHT = 300
TALL_CHART_HEIGHT = 550


def sentiment_donut(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        df,
        names="sentiment",
        values="count",
        hole=0.45,
        color="sentiment",
        color_discrete_map={"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR},
        title="Sentiment Dagilimi",
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=50, b=20), height=DEFAULT_CHART_HEIGHT)
    return fig


def model_comparison_bar(df: pd.DataFrame) -> go.Figure:
    # Metrics ordered by importance: AUC-ROC first
    metrics = ["area_under_roc", "f1_score", "accuracy", "weighted_recall", "weighted_precision"]
    metric_labels = {
        "area_under_roc": "AUC-ROC",
        "f1_score": "F1-Score",
        "accuracy": "Accuracy",
        "weighted_recall": "Recall",
        "weighted_precision": "Precision",
    }
    melted = df.melt(
        id_vars="model_name",
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = melted["metric"].map(metric_labels)
    melted["metric"] = pd.Categorical(
        melted["metric"],
        categories=["AUC-ROC", "F1-Score", "Accuracy", "Recall", "Precision"],
        ordered=True,
    )

    short_names = {
        "Logistic Regression Existing": "LR",
        "Decision Tree Classifier": "DT",
        "Random Forest Classifier": "RF",
        "Gradient Boosted Trees Classifier": "GBT",
        "Naive Bayes": "NB",
    }
    melted["model_short"] = melted["model_name"].map(lambda x: short_names.get(x, x))

    fig = px.bar(
        melted,
        x="model_short",
        y="score",
        color="metric",
        barmode="group",
        text_auto=".3f",
        title="5 Model Performans Karsilastirmasi",
        labels={"model_short": "Model", "score": "Skor", "metric": "Metrik"},
        color_discrete_sequence=MODEL_COLORS,
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_layout(
        yaxis_range=[0, 1.12],
        height=500,
        bargap=0.15,
        legend_title="Metrik",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    return fig


def confusion_matrix_heatmap(df: pd.DataFrame, model_name: str = "") -> go.Figure:
    pivot = df.pivot(index="label_text", columns="prediction_text", values="count").fillna(0)
    pivot = pivot.reindex(index=["negative", "positive"], columns=["negative", "positive"], fill_value=0)

    tn = int(pivot.loc["negative", "negative"])
    fp = int(pivot.loc["negative", "positive"])
    fn = int(pivot.loc["positive", "negative"])
    tp = int(pivot.loc["positive", "positive"])
    total = tn + fp + fn + tp

    # Quadrant color matrix: 0=TN(green), 1=FP(orange), 2=FN(red), 3=TP(blue)
    z_color = [[0, 1], [2, 3]]
    colorscale = [
        [0.000, "#27ae60"], [0.249, "#27ae60"],
        [0.250, "#f39c12"], [0.499, "#f39c12"],
        [0.500, "#e74c3c"], [0.749, "#e74c3c"],
        [0.750, "#3498db"], [1.000, "#3498db"],
    ]

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    counts = [[tn, fp], [fn, tp]]
    annotations = []
    for i in range(2):
        for j in range(2):
            pct = counts[i][j] / total * 100 if total > 0 else 0
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cell_labels[i][j]}</b><br>{counts[i][j]:,}<br>({pct:.1f}%)",
                showarrow=False,
                font=dict(color="white", size=13),
                align="center",
            ))

    fig = go.Figure(
        go.Heatmap(
            z=z_color,
            x=["Tahmin: Negative", "Tahmin: Positive"],
            y=["Gercek: Negative", "Gercek: Positive"],
            colorscale=colorscale,
            showscale=False,
            zmin=0, zmax=3,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        annotations=annotations,
        xaxis_side="bottom",
        height=DEFAULT_CHART_HEIGHT,
    )
    return fig


def feature_importance_bar(df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance") -> go.Figure:
    df = df.copy()
    df["abs_importance"] = df["importance"].abs()
    top = df.nlargest(top_n, "abs_importance")
    top = top.sort_values("abs_importance", ascending=True)

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in top["importance"]]
    max_name_len = int(top["feature"].str.len().max()) if len(top) > 0 else 20
    left_margin = max(160, max_name_len * 7)

    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            text=top["importance"].round(3),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Onem Skoru: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Onem Skoru",
        yaxis_title="Ozellik",
        height=max(DEFAULT_CHART_HEIGHT, top_n * 22),
        margin=dict(l=left_margin),
    )
    return fig


def roc_curve_chart(pred_df: pd.DataFrame) -> go.Figure:
    y_true = (pred_df["label_text"] == "positive").astype(int)
    y_score = pred_df["probability_positive"]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Youden Index: optimal threshold = max(TPR - FPR)
    youden_idx = int(np.argmax(tpr - fpr))
    opt_fpr = float(fpr[youden_idx])
    opt_tpr = float(tpr[youden_idx])
    opt_thresh = float(thresholds[youden_idx])

    # Threshold markers at 0.3, 0.5, 0.7
    thresh_markers = {}
    for t in [0.3, 0.5, 0.7]:
        closest = int(np.argmin(np.abs(thresholds - t)))
        thresh_markers[t] = (float(fpr[closest]), float(tpr[closest]))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        subplot_titles=[f"ROC Curve (AUC = {roc_auc:.4f})", "Olasilik Skoru Dagilimi"],
    )

    # Filled area between ROC curve and diagonal
    fig.add_trace(go.Scatter(
        x=np.concatenate([fpr, fpr[::-1]]),
        y=np.concatenate([tpr, fpr[::-1]]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC={roc_auc:.4f})",
        line=dict(color="#1f77b4", width=2.5),
    ), row=1, col=1)

    # Random classifier diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Rastgele Siniflandirici",
        line=dict(color="gray", width=1, dash="dash"),
    ), row=1, col=1)

    # Optimal point (Youden Index) — green star
    fig.add_trace(go.Scatter(
        x=[opt_fpr], y=[opt_tpr],
        mode="markers+text",
        name=f"Optimal (t={opt_thresh:.2f})",
        marker=dict(color="#2ecc71", size=13, symbol="star"),
        text=[f"  t={opt_thresh:.2f}"],
        textposition="middle right",
        textfont=dict(size=10),
    ), row=1, col=1)

    # Threshold markers at 0.3, 0.5, 0.7
    for t, (fx, ty) in thresh_markers.items():
        fig.add_trace(go.Scatter(
            x=[fx], y=[ty],
            mode="markers+text",
            marker=dict(color="#e67e22", size=9, symbol="circle"),
            text=[f"  t={t}"],
            textposition="middle right",
            textfont=dict(size=9),
            showlegend=False,
            hovertemplate=f"Threshold={t}<br>FPR={fx:.3f}<br>TPR={ty:.3f}<extra></extra>",
        ), row=1, col=1)

    # Probability score histogram (right panel)
    pos_scores = pred_df[pred_df["label_text"] == "positive"]["probability_positive"]
    neg_scores = pred_df[pred_df["label_text"] == "negative"]["probability_positive"]

    fig.add_trace(go.Histogram(
        x=pos_scores, name="Positive",
        marker_color=POSITIVE_COLOR, opacity=0.65, nbinsx=30,
    ), row=1, col=2)
    fig.add_trace(go.Histogram(
        x=neg_scores, name="Negative",
        marker_color=NEGATIVE_COLOR, opacity=0.65, nbinsx=30,
    ), row=1, col=2)

    fig.update_layout(
        height=TALL_CHART_HEIGHT,
        barmode="overlay",
        legend=dict(x=0.02, y=0.05),
        xaxis=dict(range=[0, 1], title="False Positive Rate"),
        yaxis=dict(range=[0, 1.02], title="True Positive Rate"),
        xaxis2=dict(title="P(positive)", range=[0, 1]),
        yaxis2=dict(title="Yorum Sayisi"),
    )
    return fig


def top_apps_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = df.nlargest(top_n, "review_count").sort_values("review_count", ascending=True)
    fig = px.bar(
        top,
        x="review_count",
        y="app_name",
        orientation="h",
        title=f"En Cok Yorum Alan {top_n} Oyun",
        labels={"review_count": "Yorum Sayisi", "app_name": "Oyun"},
        color="review_count",
        color_continuous_scale="Blues",
        text_auto=",",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=DEFAULT_CHART_HEIGHT)
    return fig


def text_length_bar(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df,
        x="sentiment",
        y="avg_review_text_length",
        color="sentiment",
        color_discrete_map={"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR},
        title="Sentiment'e Gore Ortalama Yorum Uzunlugu (karakter)",
        labels={"sentiment": "Sentiment", "avg_review_text_length": "Ort. Karakter Sayisi"},
        text_auto=".0f",
    )
    fig.update_layout(showlegend=False, height=DEFAULT_CHART_HEIGHT)
    return fig


def app_sentiment_stacked_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Stacked bar: top N apps by review count, positive/negative ratio."""
    app_counts = df.groupby(["app_name", "sentiment"]).size().unstack(fill_value=0)
    for col in ["positive", "negative"]:
        if col not in app_counts.columns:
            app_counts[col] = 0
    app_counts["total"] = app_counts["positive"] + app_counts["negative"]
    top_apps = app_counts.nlargest(top_n, "total")
    pos_pct = top_apps["positive"] / top_apps["total"] * 100
    neg_pct = top_apps["negative"] / top_apps["total"] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Positive", x=top_apps.index, y=pos_pct,
        marker_color=POSITIVE_COLOR,
        hovertemplate="%{x}<br>Positive: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Negative", x=top_apps.index, y=neg_pct,
        marker_color=NEGATIVE_COLOR,
        hovertemplate="%{x}<br>Negative: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        title=f"Top {top_n} Oyunda Sentiment Orani (%)",
        xaxis_title="Oyun",
        yaxis_title="Oran (%)",
        yaxis_range=[0, 100],
        height=DEFAULT_CHART_HEIGHT,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        xaxis_tickangle=-30,
    )
    return fig


def text_votes_scatter(df: pd.DataFrame, n_sample: int = 3000) -> go.Figure:
    """Scatter: review text length vs helpful votes, colored by sentiment."""
    df = df[df["review_votes"] >= 0].copy()
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=42)

    fig = px.scatter(
        df,
        x="text_len",
        y="review_votes",
        color="sentiment",
        color_discrete_map={"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR},
        opacity=0.45,
        title="Yorum Uzunlugu vs Helpful Votes (3k ornek, Silver verisi)",
        labels={
            "text_len": "Yorum Uzunlugu (karakter)",
            "review_votes": "Helpful Votes",
            "sentiment": "Sentiment",
        },
    )
    # Manual trendline via numpy polyfit
    if len(df) >= 2:
        x_arr = df["text_len"].values
        y_arr = df["review_votes"].values
        m, b = np.polyfit(x_arr, y_arr, 1)
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_fit, y=m * x_fit + b,
            mode="lines",
            name="Trend",
            line=dict(color="black", width=2, dash="dot"),
        ))
    fig.update_layout(height=DEFAULT_CHART_HEIGHT)
    return fig
