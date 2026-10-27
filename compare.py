"""Print the benchmark table from results/*.json."""
import json

from config import CFG

NAMES = ["tfidf_logreg", "distilbert_raw", "distilbert_clean"]
rows = {}
for n in NAMES:
    p = CFG.results_dir / f"{n}.json"
    if p.exists():
        rows[n] = json.load(open(p))

print(f"{'model':<20}{'acc_all':>9}{'f1_all':>9}{'acc_ment':>10}{'f1_ment':>9}")
for n, r in rows.items():
    print(f"{n:<20}{r['acc_all']:>9.3f}{r['f1_all']:>9.3f}{r['acc_mentioned']:>10.3f}{r['f1_mentioned']:>9.3f}")

if "distilbert_raw" in rows and "distilbert_clean" in rows:
    d = 100 * (rows["distilbert_clean"]["f1_all"] - rows["distilbert_raw"]["f1_all"])
    print(f"\nPreprocessing effect on macro-F1 (clean - raw): {d:+.1f} points")
if "tfidf_logreg" in rows and "distilbert_clean" in rows:
    d = 100 * (rows["distilbert_clean"]["acc_all"] - rows["tfidf_logreg"]["acc_all"])
    print(f"DistilBERT vs TF-IDF+LR accuracy gap: {d:+.1f} points")
