import json
import unicodedata

from tqdm import tqdm

from garumzime.config import Config
from garumzime.constants import DATASETS_PATH, PROCESSED_DATA_PATH


def _remove_diacritics(input_str):
    # Normalize to NFD (decompose characters into base + combining marks)
    nfkd_form = unicodedata.normalize("NFD", input_str)
    # Filter out combining marks (category 'Mn')
    return "".join([c for c in nfkd_form if unicodedata.category(c) != "Mn"])


def prepare_data(cfg: Config):
    dataset_path = f"{DATASETS_PATH / cfg.dataset_name}.jsonl"
    out_path = f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl"

    seq_len: int = cfg.seq_length

    all_chars = set()

    with open(dataset_path, "rb") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(list(in_f), desc="Processing documents..."):
            text = json.loads(line)["text"]
            filtered_text = _remove_diacritics(text)
            # dumb splitting - every "seq_len" characters is split
            # TODO something better
            for i in range(-(len(text) // -seq_len)):  # dirty upside-down division
                feature = text[i * seq_len : (i + 1) * seq_len]
                target = filtered_text[i * seq_len : (i + 1) * seq_len]
                out = {"feature": feature, "target": target}
                out_f.write(json.dumps(out) + "\n")
                chars = set(feature)
                all_chars |= chars

    print(all_chars)
    all_chars_list = list(all_chars)
    with open("tokens.json", "w") as final:
        json.dump(all_chars_list, final)


if __name__ == "__main__":
    cfg = Config.from_toml()
    prepare_data(cfg)
