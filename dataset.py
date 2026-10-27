import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ASPECTS, L2I
from preprocess import clean_text


def load_splits(path, seed: int = 42, clean: bool = True):
    """80/10/10 split. Returns {split: (texts, labels[N, n_aspects])}. Same seed -> same split for every model."""
    df = pd.read_csv(path).dropna(subset=["text"]).reset_index(drop=True)
    y = np.stack([df[a].fillna("not_mentioned").map(L2I).values for a in ASPECTS], axis=1)
    texts = df["text"].astype(str).map(clean_text if clean else str.strip).tolist()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed)
    pick = lambda ix: ([texts[i] for i in ix], y[ix])
    return {"train": pick(tr), "val": pick(va), "test": pick(te)}


class ReviewDataset:
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len):
    import torch
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(np.stack(labels), dtype=torch.long)
        return enc
    return collate
