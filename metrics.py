import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import ASPECTS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_*: [N, n_aspects] ints in {0 not_mentioned, 1 neg, 2 neu, 3 pos}."""
    t, p = y_true.reshape(-1), y_pred.reshape(-1)
    mask = t > 0
    out = {
        "acc_all": accuracy_score(t, p),
        "f1_all": f1_score(t, p, average="macro", labels=[0, 1, 2, 3], zero_division=0),
        "acc_mentioned": accuracy_score(t[mask], p[mask]),
        "f1_mentioned": f1_score(t[mask], p[mask], average="macro", labels=[1, 2, 3], zero_division=0),
        "per_aspect_f1": {},
    }
    for j, a in enumerate(ASPECTS):
        out["per_aspect_f1"][a] = f1_score(y_true[:, j], y_pred[:, j], average="macro", labels=[0, 1, 2, 3], zero_division=0)
    return out
