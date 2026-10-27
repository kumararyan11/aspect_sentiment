"""Create the aspect-labelled dataset (data/reviews.csv).

Columns: text, price, quality, delivery, packaging, customer_service, overall_experience
Values : not_mentioned | negative | neutral | positive

Two modes:
  synthetic : template-generated noisy reviews (for testing the pipeline end to end)
  weak      : weak-label a REAL reviews CSV (text + star rating) with aspect keywords + a sentiment lexicon
              (use hand-corrected labels for a subset if you can - weak labels cap achievable accuracy)
"""
import argparse
import random
import re

import pandas as pd

from config import ASPECTS, CFG
from preprocess import clean_text

# ------------------------------------------------------------------ synthetic
T = {
    "price": {
        "positive": ["Totally worth the price", "great value for money", "very affordable for this quality", "paisa vasool!!", "priced really well vfm"],
        "negative": ["way too expensive for what it is", "overpriced, not worth the money", "mehenga hai bahut", "price is a rip off", "cost too much honestly"],
        "neutral": ["The price is okay, nothing special", "average pricing compared to others", "price is fair I guess"],
    },
    "quality": {
        "positive": ["build quality is excellent", "sturdy and durable, works perfectly", "the quality is top notch", "material feels premium", "gr8 quality"],
        "negative": ["quality is poor, broke in a week", "stopped working after two days, defective", "ghatiya quality, feels flimsy", "the material is cheap and stitching came apart", "bad quality bakwas"],
        "neutral": ["quality is average", "build is okay, nothing great", "works fine but nothing special in quality"],
    },
    "delivery": {
        "positive": ["delivery was super fast", "arrived a day early", "delivered on time, very quick shipping", "fast dlvry thx", "courier was prompt"],
        "negative": ["delivery was very late", "shipping took forever, delayed by a week", "courier delayed and never updated", "product arrived late again", "delivery was a mess and slow"],
        "neutral": ["delivery took the usual time", "arrived on the expected date", "shipping was average"],
    },
    "packaging": {
        "positive": ["packaging was excellent and secure", "well packed with proper bubble wrap", "neatly packed box", "pkging was great"],
        "negative": ["packaging was damaged and torn", "box was crushed, poorly packed", "terrible packaging, item almost broke", "pkg was bad, seal open"],
        "neutral": ["packaging was normal", "regular box, nothing special in packaging", "packaging is okay"],
    },
    "customer_service": {
        "positive": ["customer service was very helpful", "support resolved my issue quickly", "seller replaced it without any fuss", "great support team, refund was fast"],
        "negative": ["customer service was rude and unhelpful", "support never replied to my emails", "seller refused the refund, worst service", "no response from customer care"],
        "neutral": ["customer service replied after a day", "support was okay", "service was average"],
    },
    "overall_experience": {
        "positive": ["overall very happy with this purchase", "totally recommend it", "overall a great experience", "loved it, will buy again", "satisfied overall"],
        "negative": ["overall a disappointing experience", "would not recommend to anyone", "total waste, very disappointed", "worst purchase ever, never again"],
        "neutral": ["overall it is an okay purchase", "mixed experience overall", "it is fine, I have no strong feelings"],
    },
}
FILLER = ["Received the {p} yesterday.", "Bought this {p} last week.", "Using the {p} for a month now.", "Ordered the {p} for my brother."]
PRODUCTS = ["phone", "headphones", "blender", "backpack", "shoes", "keyboard", "watch", "kettle"]
EMO = {"positive": ["😍", "👍", "🔥", "😊", ":)"], "negative": ["😡", "👎", "😞", "🤬", ":("], "neutral": ["😐", "🙄"]}


def _noise(s: str, sent: str) -> str:
    r = random.random()
    if r < 0.12:
        s = re.sub(r"\b(good|great|late|bad|slow|fast)\b", lambda m: m.group(1) + m.group(1)[-1] * 3, s)
    if r > 0.7:
        s += " " + random.choice(EMO[sent])
    if random.random() < 0.15:
        s = s.upper() if random.random() < 0.3 else s.lower()
    if random.random() < 0.2:
        s += random.choice(["!!!", "!!", "...", "?!"])
    return s


def synthetic(n: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _ in range(n):
        labels = {a: "not_mentioned" for a in ASPECTS}
        parts = []
        if random.random() < 0.5:
            parts.append(random.choice(FILLER).format(p=random.choice(PRODUCTS)))
        for a in random.sample(ASPECTS, random.choice([1, 2, 2, 3, 4])):
            sent = random.choices(["positive", "negative", "neutral"], [0.5, 0.35, 0.15])[0]
            parts.append(_noise(random.choice(T[a][sent]), sent) + random.choice([".", "", ","]))
            labels[a] = sent
        random.shuffle(parts)
        rows.append({"text": " ".join(parts), **labels})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ weak labelling
KEYWORDS = {
    "price": ["price", "cost", "expensive", "cheap", "value for money", "worth", "money", "overpriced", "affordable", "pricey", "mrp", "discount"],
    "quality": ["quality", "build", "material", "durable", "broke", "broken", "stitching", "defective", "sturdy", "flimsy", "working", "works", "stopped working", "performance"],
    "delivery": ["delivery", "delivered", "shipping", "shipped", "arrived", "courier", "late", "delay", "delayed", "dispatch", "quickly"],
    "packaging": ["packaging", "packed", "package", "box", "wrapped", "wrap", "sealed", "seal"],
    "customer_service": ["customer service", "customer care", "support", "seller", "refund", "replacement", "return", "helpline", "response", "service"],
}
POS = {"good", "great", "excellent", "amazing", "love", "perfect", "fast", "quick", "quickly", "worth", "sturdy", "durable", "nice", "awesome", "helpful", "best", "happy", "satisfied", "affordable", "premium", "value for money", "early", "prompt", "secure", "neat", "neatly"}
NEG = {"bad", "poor", "worst", "terrible", "late", "delay", "delayed", "broken", "broke", "damaged", "defective", "expensive", "overpriced", "useless", "rude", "disappointed", "disappointing", "flimsy", "waste", "fraud", "fake", "slow", "horrible", "pathetic", "refused", "torn", "crushed", "rubbish", "worthless", "angry", "sad", "unhappy"}
NEGATE = {"not", "no", "never", "cannot"}


def _sentence_sentiment(sent: str) -> str:
    toks = re.findall(r"[a-z']+", sent)
    score = 0
    for i, t in enumerate(toks):
        s = 1 if t in POS else -1 if t in NEG else 0
        if s and any(x in NEGATE for x in toks[max(0, i - 2):i]):
            s = -s
        score += s
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def weak_label(df: pd.DataFrame, text_col: str, rating_col: str) -> pd.DataFrame:
    rows = []
    for text, rating in zip(df[text_col], df[rating_col]):
        labels = {a: "not_mentioned" for a in ASPECTS}
        sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+| but ", clean_text(text)) if s]
        for a, kws in KEYWORDS.items():
            hits = [s for s in sents if any(re.search(rf"\b{re.escape(k)}\b", s) for k in kws)]
            if hits:
                votes = [_sentence_sentiment(s) for s in hits]
                labels[a] = max(set(votes), key=votes.count)
        r = float(rating)
        labels["overall_experience"] = "positive" if r >= 4 else "neutral" if r == 3 else "negative"
        rows.append({"text": text, **labels})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--n", type=int, default=8000)
    w = sub.add_parser("weak"); w.add_argument("--input", required=True); w.add_argument("--text_col", default="review"); w.add_argument("--rating_col", default="rating")
    for p in (s, w):
        p.add_argument("--out", default=str(CFG.data_path))
    a = ap.parse_args()
    df = synthetic(a.n) if a.mode == "synthetic" else weak_label(pd.read_csv(a.input), a.text_col, a.rating_col)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {len(df)} rows to {a.out}")
    for c in ASPECTS:
        print(c, df[c].value_counts().to_dict())
