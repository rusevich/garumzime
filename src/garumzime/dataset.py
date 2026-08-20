import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from garumzime.config import Config
from garumzime.constants import PROCESSED_DATA_PATH, TOKENIZER_PATH


class DiacriticDataset(Dataset):
    def __init__(self, cfg: Config, shuffle: bool):
        with open(f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl") as f:
            self.data = [json.loads(line) for line in f]
        with open(f"{TOKENIZER_PATH / cfg.tokenizer_name}.json") as f:
            vocab = json.load(f)["vocab"]
        self.vocab = {ch: int(v["id"]) for ch, v in vocab.items()}
        self.unk_id = self.vocab["<unk>"]
        self.pad_id = self.vocab["<pad>"]
        self.seq_len = cfg.seq_length

    def __len__(self):
        return len(self.data)

    def _encode(self, s):
        get = self.vocab.get
        return torch.tensor([get(ch, self.unk_id) for ch in s[: self.seq_len]], dtype=torch.long)

    def __getitem__(self, idx):
        rec = self.data[idx]
        return self._encode(rec["feature"]), self._encode(rec["target"])

    def collate(self, batch):
        # feature, target, mask
        feats, tgts = zip(*batch, strict=True)
        f = pad_sequence(feats, batch_first=True, padding_value=self.pad_id)
        t = pad_sequence(tgts, batch_first=True, padding_value=self.pad_id)
        return f, t, f != self.pad_id
