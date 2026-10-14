from __future__ import annotations

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

from .preprocessing import clean_text


SENTIMENT_THRESHOLDS = {
    "positive": 0.05,
    "negative": -0.05,
}


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def sentiment_label(compound: float) -> str:
    if compound >= SENTIMENT_THRESHOLDS["positive"]:
        return "Positive"
    if compound <= SENTIMENT_THRESHOLDS["negative"]:
        return "Negative"
    return "Neutral"


def get_emotions(text: str) -> dict:
    """Return NRC raw emotion counts and dominant emotion."""
    if not text.strip():
        return {
            "anger": 0, "anticipation": 0, "disgust": 0, "fear": 0,
            "joy": 0, "sadness": 0, "surprise": 0, "trust": 0,
            "dominant_emotion": "none"
        }

    emotion_obj = NRCLex(text)
    raw = getattr(emotion_obj, "raw_emotion_scores", None)

    if raw is None:
        raw = getattr(emotion_obj, "affect_frequencies", {})

    core_emotions = [
        "anger", "anticipation", "disgust", "fear",
        "joy", "sadness", "surprise", "trust"
    ]
    result = {emotion: int(raw.get(emotion, 0)) for emotion in core_emotions}

    if sum(result.values()) == 0:
        dominant = "none"
    else:
        dominant = max(result, key=result.get)

    result["dominant_emotion"] = dominant
    return result


def analyze_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Analyze one source DataFrame. Requires a 'text' column."""
    ensure_nltk_resources()

    if "text" not in df.columns:
        raise ValueError("Input data must contain a 'text' column.")

    analyzer = SentimentIntensityAnalyzer()
    result = df.copy()
    result["source"] = source_name
    result["clean_text"] = result["text"].apply(clean_text)

    scores = result["clean_text"].apply(analyzer.polarity_scores).apply(pd.Series)
    scores = scores.rename(columns={
        "neg": "negative_score",
        "neu": "neutral_score",
        "pos": "positive_score",
        "compound": "compound_score",
    })

    result = pd.concat([result, scores], axis=1)
    result["sentiment"] = result["compound_score"].apply(sentiment_label)

    emotion_df = result["clean_text"].apply(get_emotions).apply(pd.Series)
    result = pd.concat([result, emotion_df], axis=1)

    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")

    return result
