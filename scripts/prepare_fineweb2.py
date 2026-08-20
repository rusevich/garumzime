import json

from datasets import load_dataset

from garumzime.config import Config
from garumzime.constants import LOCAL_DATA
from garumzime.prepare_data import prepare_data

OUT = LOCAL_DATA / "datasets" / "fineweb2.jsonl"
LIMIT = 5000


def download():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2", name="lvs_Latn", split="train", streaming=True
    ).take(LIMIT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    cfg = Config.from_toml()
    prepare_data(cfg)


if __name__ == "__main__":
    download()
