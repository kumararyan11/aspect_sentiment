import json

import torch
from transformers import AutoTokenizer

from config import ASPECTS, CFG, LABELS
from model import MultiAspectSentiment
from preprocess import clean_text


class AspectAnalyzer:
    def __init__(self, run: str = "distilbert_clean"):
        d = CFG.outputs_dir / run
        meta = json.load(open(d / "meta.json"))
        self.meta = meta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        self.model = MultiAspectSentiment(meta["model_name"], meta["dropout"])
        self.model.load_state_dict(torch.load(d / "model.pt", map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def analyze(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            proc = [clean_text(t) if self.meta["clean"] else t for t in chunk]
            enc = self.tok(proc, truncation=True, max_length=self.meta["max_len"], padding=True, return_tensors="pt").to(self.device)
            probs = self.model(enc["input_ids"], enc["attention_mask"]).softmax(-1).cpu()
            for p in probs:
                out.append({a: {"label": LABELS[int(p[j].argmax())], "confidence": float(p[j].max())} for j, a in enumerate(ASPECTS)})
        return out


if __name__ == "__main__":
    import sys
    r = AspectAnalyzer().analyze([" ".join(sys.argv[1:]) or "Great quality 😍 but delivery was sooo late and packaging was damaged"])[0]
    for a, v in r.items():
        print(f"{a:<20}{v['label']:<15}{v['confidence']:.2f}")
