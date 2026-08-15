import json

from garumzime.config import Config
from garumzime.constants import PROCESSED_DATA_PATH, TOKENIZER_PATH


class Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_path = f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl"
        self.tokenizer_path = f"{TOKENIZER_PATH / cfg.tokenizer_name}.json"
        self.data = []
        self.tok = {}
        with open(self.data_path) as jsonfile:
            for line in jsonfile:
                self.data.append(json.loads(line))
        with open(self.tokenizer_path) as tokfile:
            self.tok = json.load(tokfile)
        self.seq_len = cfg.seq_length

        self.unk_id = int(self.tok["vocab"]["<unk>"]["id"])
        self.pad_id = int(self.tok["vocab"]["<pad>"]["id"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]["feature"]
        target = self.data[idx]["target"]

        feature_ids, target_ids = [], []
        for f, t in zip(feature, target, strict=True):
            f_id, t_id = self.unk_id, self.unk_id
            if f in self.tok["vocab"]:
                f_id = int(self.tok["vocab"][f]["id"])
            if t in self.tok["vocab"]:
                t_id = int(self.tok["vocab"][t]["id"])
            feature_ids.append(f_id)
            target_ids.append(t_id)

        # must be the same for target and feature
        padding = [self.pad_id] * (self.seq_len - len(feature_ids))

        feature_ids += padding
        target_ids += padding

        return feature_ids, target_ids
