from pathlib import Path
import pandas as pd

from src.analyzer import analyze_dataframe
from src.reporting import (
    build_summary,
    build_emotion_summary,
    generate_insights,
    save_sentiment_chart,
    save_source_chart,
    save_trend_chart,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_and_analyze() -> pd.DataFrame:
    sources = {
        "Amazon Reviews": DATA_DIR / "amazon_reviews.csv",
        "Social Media": DATA_DIR / "social_media.csv",
        "News": DATA_DIR / "news.csv",
    }

    frames = []
    for source_name, path in sources.items():
        df = pd.read_csv(path)
        analyzed = analyze_dataframe(df, source_name)
        frames.append(analyzed)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyzed = load_and_analyze()
    analyzed.to_csv(OUTPUT_DIR / "analyzed_text.csv", index=False)

    summary = build_summary(analyzed)
    summary.to_csv(OUTPUT_DIR / "sentiment_summary.csv", index=False)

    emotion_summary = build_emotion_summary(analyzed)
    emotion_summary.to_csv(OUTPUT_DIR / "emotion_summary.csv", index=False)

    insights = generate_insights(analyzed)
    (OUTPUT_DIR / "insights.txt").write_text(
        "\n".join(f"- {item}" for item in insights),
        encoding="utf-8",
    )

    save_sentiment_chart(analyzed, OUTPUT_DIR)
    save_source_chart(analyzed, OUTPUT_DIR)
    save_trend_chart(analyzed, OUTPUT_DIR)

    print("\nSentiment Analysis Completed")
    print("=" * 35)
    print(summary.to_string(index=False))
    print("\nKey Insights:")
    for item in insights:
        print(f"- {item}")
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
