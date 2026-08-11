from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA = ROOT / ".local_data"

LOCAL_DATA.mkdir(exist_ok=True)
