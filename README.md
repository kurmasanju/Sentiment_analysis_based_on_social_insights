# Sentiment Analysis & Emotion Detection Project

## Project Objective

This project analyzes text and classifies it as **Positive, Negative, or Neutral**, detects specific emotions using NLP lexicons, compares sentiment patterns across multiple source types, and summarizes public opinion for business or social insights.

It covers the requested task points:

1. Classify text as positive, negative, or neutral.
2. Detect emotions using NLP techniques and lexicons.
3. Analyze Amazon-review-style, social-media-style, and news-style text.
4. Identify public opinion and trends through sentiment patterns.
5. Turn the results into marketing, product-development, and social insights.

## Project Structure

```text
sentiment_analysis_project/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── amazon_reviews.csv
│   ├── social_media.csv
│   └── news.csv
├── outputs/
└── src/
    ├── __init__.py
    ├── analyzer.py
    ├── preprocessing.py
    ├── reporting.py
    └── utils.py
```

## Methodology

### Sentiment Analysis
The project uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** from NLTK.

VADER returns:
- positive score
- neutral score
- negative score
- compound score

Classification rule:
- Compound >= 0.05 -> Positive
- Compound <= -0.05 -> Negative
- Otherwise -> Neutral

### Emotion Detection
The project uses **NRC Emotion Lexicon** through `NRCLex`.

Supported emotion categories include:
- joy
- trust
- anticipation
- surprise
- sadness
- anger
- fear
- disgust

### Trend Analysis
The project aggregates sentiment by:
- source
- date
- sentiment class
- dominant emotion

It also produces useful business recommendations.

## How to Run

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python main.py
```

Results will be generated inside the `outputs/` folder.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Replacing the Demo Data

Each CSV should contain at minimum:

```text
date,text
```

You may optionally include extra columns such as:

```text
rating,product,category,author
```

Keep the filenames or update `main.py` and `app.py` to point to your own files.

## Practical Applications

### Marketing
- Find messages and product features associated with positive sentiment.
- Detect campaigns or topics producing negative reactions.
- Track public response over time.

### Product Development
- Identify complaints and negative emotions.
- Discover frequently praised product qualities.
- Prioritize improvements based on recurring issues.

### Social Insights
- Track public mood.
- Detect shifts in fear, anger, trust, or joy.
- Compare reactions across channels and topics.

## Notes

The included CSV files are small demo datasets designed to make the project run immediately. For a real academic or business project, replace them with a larger licensed/public dataset.
