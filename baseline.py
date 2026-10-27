"""TF-IDF + Logistic Regression baseline (one classifier per aspect), same split as DistilBERT."""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import ASPECTS, CFG
from dataset import load_splits
from metrics import compute_metrics
from preprocess import custom_tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CFG.data_path))
    a = ap.parse_args()

    s = load_splits(a.data, CFG.seed, clean=True)
    (Xtr, ytr), (Xte, yte) = s["train"], s["test"]

    vec = TfidfVectorizer(tokenizer=custom_tokenize, token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    Vtr, Vte = vec.fit_transform(Xtr), vec.transform(Xte)

    pred = np.zeros_like(yte)
    for j, asp in enumerate(ASPECTS):
        clf = LogisticRegression(max_iter=1000, C=4.0)
        clf.fit(Vtr, ytr[:, j])
        pred[:, j] = clf.predict(Vte)

    res = compute_metrics(yte, pred)
    res.update(model="tfidf_logreg", n_train=len(Xtr), n_test=len(Xte))
    print(json.dumps(res, indent=2))
    CFG.results_dir.mkdir(exist_ok=True)
    json.dump(res, open(CFG.results_dir / "tfidf_logreg.json", "w"), indent=2)


if __name__ == "__main__":
    main()
