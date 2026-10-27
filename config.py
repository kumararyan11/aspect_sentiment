from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent

ASPECTS = ["price", "quality", "delivery", "packaging", "customer_service", "overall_experience"]
LABELS = ["not_mentioned", "negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}


@dataclass
class Config:
    data_path: Path = ROOT / "data" / "reviews.csv"
    results_dir: Path = ROOT / "results"
    outputs_dir: Path = ROOT / "outputs"

    model_name: str = "distilbert-base-uncased"   # use "distilbert-base-multilingual-cased" for heavy multilingual data
    max_len: int = 128
    batch_size: int = 32
    epochs: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    dropout: float = 0.2
    patience: int = 2
    seed: int = 42


CFG = Config()
