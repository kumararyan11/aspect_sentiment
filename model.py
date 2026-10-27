import torch
import torch.nn as nn
from transformers import AutoModel

from config import ASPECTS, LABELS


class MultiAspectSentiment(nn.Module):
    """Shared DistilBERT encoder + one 4-way head per aspect (not_mentioned / neg / neutral / pos)."""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h, len(ASPECTS) * len(LABELS))

    def forward(self, input_ids, attention_mask):
        hid = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (hid * m).sum(1) / m.sum(1).clamp(min=1e-9)   # mean pooling
        return self.head(self.dropout(pooled)).view(-1, len(ASPECTS), len(LABELS))
