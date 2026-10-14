from pathlib import Path
import pandas as pd
import streamlit as st

from src.analyzer import analyze_dataframe
from src.reporting import build_summary, build_emotion_summary, generate_insights

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Sentiment Analysis & Emotion Detection")
st.caption("Positive / Negative / Neutral classification + NRC emotion analysis")

@st.cache_data
def load_default_data():
    sources = {
        "Amazon Reviews": DATA_DIR / "amazon_reviews.csv",
        "Social Media": DATA_DIR / "social_media.csv",
        "News": DATA_DIR / "news.csv",
    }

    frames = []
    for source_name, path in sources.items():
        df = pd.read_csv(path)
        frames.append(analyze_dataframe(df, source_name))
    return pd.concat(frames, ignore_index=True)

uploaded_file = st.sidebar.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

if uploaded_file is not None:
    custom_df = pd.read_csv(uploaded_file)
    if "text" not in custom_df.columns:
        st.error("The uploaded CSV must contain a column named 'text'.")
        st.stop()
    data = analyze_dataframe(custom_df, "Uploaded Data")
else:
    data = load_default_data()

source_options = ["All"] + sorted(data["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Filter by source", source_options)

if selected_source != "All":
    filtered = data[data["source"] == selected_source].copy()
else:
    filtered = data.copy()

st.subheader("Overview")
counts = filtered["sentiment"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Texts", len(filtered))
c2.metric("Positive", int(counts.get("Positive", 0)))
c3.metric("Neutral", int(counts.get("Neutral", 0)))
c4.metric("Negative", int(counts.get("Negative", 0)))

st.subheader("Sentiment Distribution")
sentiment_counts = (
    filtered["sentiment"]
    .value_counts()
    .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
)
st.bar_chart(sentiment_counts)

st.subheader("Emotion Distribution")
emotion_cols = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]
emotion_totals = filtered[emotion_cols].sum().sort_values(ascending=False)
st.bar_chart(emotion_totals)

if "date" in filtered.columns and filtered["date"].notna().any():
    st.subheader("Sentiment Trend Over Time")
    trend = (
        filtered.dropna(subset=["date"])
        .assign(day=lambda x: x["date"].dt.date)
        .groupby("day")["compound_score"]
        .mean()
    )
    st.line_chart(trend)

st.subheader("Business / Social Insights")
for insight in generate_insights(filtered):
    st.write("•", insight)

st.subheader("Analyzed Data")
columns_to_show = [
    c for c in [
        "date", "source", "text", "sentiment", "compound_score", "dominant_emotion"
    ] if c in filtered.columns
]
st.dataframe(filtered[columns_to_show], use_container_width=True)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Analyzed CSV",
    data=csv,
    file_name="sentiment_results.csv",
    mime="text/csv",
)
