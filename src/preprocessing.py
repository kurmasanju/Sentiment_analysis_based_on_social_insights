import re
import pandas as pd


def clean_text(text: str) -> str:
    """Basic text cleaning while preserving sentiment-bearing punctuation."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"@\\w+", " ", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text
