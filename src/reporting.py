from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    grouped = (
        df.groupby(["source", "sentiment"])
          .size()
          .reset_index(name="count")
    )
    source_totals = grouped.groupby("source")["count"].transform("sum")
    grouped["percentage"] = (grouped["count"] / source_totals * 100).round(2)
    return grouped


def build_emotion_summary(df: pd.DataFrame) -> pd.DataFrame:
    emotion_cols = [
        "anger", "anticipation", "disgust", "fear",
        "joy", "sadness", "surprise", "trust"
    ]
    available = [c for c in emotion_cols if c in df.columns]
    rows = []
    for source, group in df.groupby("source"):
        totals = group[available].sum().sort_values(ascending=False)
        for emotion, count in totals.items():
            rows.append({"source": source, "emotion": emotion, "count": int(count)})
    return pd.DataFrame(rows)


def generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    if df.empty:
        return ["No data available for analysis."]

    overall = df["sentiment"].value_counts(normalize=True).mul(100)
    pos = overall.get("Positive", 0.0)
    neg = overall.get("Negative", 0.0)
    neu = overall.get("Neutral", 0.0)

    insights.append(
        f"Overall sentiment is {pos:.1f}% positive, {neg:.1f}% negative, and {neu:.1f}% neutral."
    )

    if pos > neg:
        insights.append(
            "Public opinion is more favorable than unfavorable. Marketing can reinforce the themes found in positive comments."
        )
    elif neg > pos:
        insights.append(
            "Negative sentiment is stronger than positive sentiment. Product or communication teams should investigate recurring complaints."
        )
    else:
        insights.append(
            "Positive and negative sentiment are balanced. More context or a larger dataset may be needed before making major decisions."
        )

    negative_rows = df[df["sentiment"] == "Negative"]
    if not negative_rows.empty:
        common_negative_emotion = negative_rows["dominant_emotion"].value_counts().idxmax()
        insights.append(
            f"The most common dominant emotion among negative texts is '{common_negative_emotion}'. "
            "This can help teams distinguish frustration, fear, sadness, or other concerns."
        )

    for source, group in df.groupby("source"):
        breakdown = group["sentiment"].value_counts(normalize=True).mul(100)
        source_pos = breakdown.get("Positive", 0.0)
        source_neg = breakdown.get("Negative", 0.0)
        insights.append(
            f"{source}: {source_pos:.1f}% positive and {source_neg:.1f}% negative."
        )

    if "date" in df.columns and df["date"].notna().any():
        daily = (
            df.dropna(subset=["date"])
              .assign(day=lambda x: x["date"].dt.date)
              .groupby("day")["compound_score"]
              .mean()
        )
        if len(daily) >= 2:
            direction = "improved" if daily.iloc[-1] > daily.iloc[0] else "declined"
            insights.append(
                f"Average sentiment {direction} from the first to the last date in this sample."
            )

    return insights


def save_sentiment_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "sentiment_distribution.png"

    counts = df["sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )
    ax = counts.plot(kind="bar", title="Overall Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Texts")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


def save_source_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "sentiment_by_source.png"

    table = pd.crosstab(df["source"], df["sentiment"])
    table = table.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    ax = table.plot(kind="bar", title="Sentiment by Source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Number of Texts")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


def save_trend_chart(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "date" not in df.columns or not df["date"].notna().any():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "sentiment_trend.png"

    trend = (
        df.dropna(subset=["date"])
          .assign(day=lambda x: x["date"].dt.date)
          .groupby("day")["compound_score"]
          .mean()
    )

    ax = trend.plot(marker="o", title="Average Sentiment Trend")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Compound Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path
