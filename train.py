"""Fine-tune DistilBERT for multi-aspect sentiment.

python train.py --data data/reviews.csv                # cleaned text (proposed)
python train.py --data data/reviews.csv --no_clean     # raw text ablation (to measure the F1 gain from preprocessing)
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import ASPECTS, CFG, LABELS
from dataset import ReviewDataset, load_splits, make_collate
from metrics import compute_metrics
from model import MultiAspectSentiment


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        ps.append(logits.argmax(-1).cpu().numpy()); ys.append(b["labels"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CFG.data_path))
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--model_name", default=CFG.model_name)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    a = ap.parse_args()

    set_seed(CFG.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = "distilbert_raw" if a.no_clean else "distilbert_clean"
    run_dir = CFG.outputs_dir / run
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(a.data, CFG.seed, clean=not a.no_clean)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    collate = make_collate(tok, CFG.max_len)
    loaders = {
        k: DataLoader(ReviewDataset(*v), batch_size=a.batch_size, shuffle=(k == "train"), collate_fn=collate)
        for k, v in splits.items()
    }

    # class weights (sqrt inverse frequency) to cope with the many not_mentioned labels
    counts = np.bincount(splits["train"][1].reshape(-1), minlength=len(LABELS)).astype(float) + 1
    w = (counts.sum() / (len(LABELS) * counts)) ** 0.5
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float, device=device))

    model = MultiAspectSentiment(a.model_name, CFG.dropout).to(device)
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=CFG.weight_decay)
    total = len(loaders["train"]) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(CFG.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"epoch {ep}/{a.epochs}")
        for b in pbar:
            ids, mask, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(ids, mask)
                loss = loss_fn(logits.reshape(-1, len(LABELS)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        yv, pv = evaluate(model, loaders["val"], device)
        m = compute_metrics(yv, pv)
        print(f"  val acc_all={m['acc_all']:.4f} f1_all={m['f1_all']:.4f} f1_mentioned={m['f1_mentioned']:.4f}")
        if m["f1_all"] > best:
            best, bad = m["f1_all"], 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            bad += 1
            if bad >= CFG.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    yt, pt = evaluate(model, loaders["test"], device)
    res = compute_metrics(yt, pt)
    res.update(model=run, n_train=len(splits["train"][0]), n_test=len(splits["test"][0]))
    print(json.dumps(res, indent=2))

    tok.save_pretrained(run_dir / "tokenizer")
    json.dump({"model_name": a.model_name, "max_len": CFG.max_len, "clean": not a.no_clean, "dropout": CFG.dropout}, open(run_dir / "meta.json", "w"))
    CFG.results_dir.mkdir(exist_ok=True)
    json.dump(res, open(CFG.results_dir / f"{run}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
