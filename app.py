import pandas as pd
import streamlit as st

from config import ASPECTS, LABELS
from predict import AspectAnalyzer

st.set_page_config(page_title="Aspect-Based Review Sentiment", page_icon="🛒", layout="wide")
st.title("🛒 Aspect-Based Sentiment Analysis of Product Reviews")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return AspectAnalyzer()


model = get_model()
tab1, tab2 = st.tabs(["Single review", "Batch (CSV)"])

with tab1:
    text = st.text_area("Review", "Quality is gr8 😍 but delivery was sooo late and the box was damaged. Paisa vasool though!")
    if st.button("Analyze") and text:
        res = model.analyze([text])[0]
        st.dataframe(pd.DataFrame([{"aspect": a, "sentiment": v["label"], "confidence": round(v["confidence"], 3)} for a, v in res.items()]), use_container_width=True)

with tab2:
    f = st.file_uploader("CSV with a 'text' column", type="csv")
    if f:
        df = pd.read_csv(f)
        preds = model.analyze(df["text"].astype(str).tolist())
        for a in ASPECTS:
            df[a] = [p[a]["label"] for p in preds]
        st.dataframe(df.head(100), use_container_width=True)
        summary = pd.DataFrame({a: df[a].value_counts() for a in ASPECTS}).reindex(LABELS).fillna(0).astype(int)
        st.bar_chart(summary.drop(index="not_mentioned").T)
        st.download_button("Download results", df.to_csv(index=False), "aspect_sentiment.csv")
