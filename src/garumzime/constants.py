import os
from pathlib import Path

ROOT = Path(os.environ.get("GARUMZIME_HOME", Path.cwd())).resolve()

LOCAL_DATA = ROOT / ".local_data"
DATASETS_PATH = LOCAL_DATA / "datasets"
PROCESSED_DATA_PATH = LOCAL_DATA / "processed"
TOKENIZER_PATH = LOCAL_DATA / "tokenizers"

CONFIG_PATH = Path(os.environ.get("GARUMZIME_CONFIG", ROOT / "garumzime-config.toml"))

for p in (LOCAL_DATA, DATASETS_PATH, PROCESSED_DATA_PATH, TOKENIZER_PATH):
    p.mkdir(parents=True, exist_ok=True)
