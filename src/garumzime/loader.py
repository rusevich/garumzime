import json

from garumzime.config import Config
from garumzime.constants import PROCESSED_DATA_PATH


class Loader:
    def __init__(self, cfg: Config):
        self.data_path = f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl"
        self.data = []
        with open(self.data_path) as jsonfile:
            for line in jsonfile:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["feature"], self.data[idx]["target"]
