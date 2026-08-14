from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOCAL_DATA = ROOT / ".local_data"
DATASETS_PATH = LOCAL_DATA / "datasets"
PROCESSED_DATA_PATH = LOCAL_DATA / "processed"

CONFIG_PATH = ROOT / "garumzime-config.toml"

LOCAL_DATA.mkdir(exist_ok=True)
DATASETS_PATH.mkdir(exist_ok=True)
PROCESSED_DATA_PATH.mkdir(exist_ok=True)
