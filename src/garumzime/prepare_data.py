import json
import unicodedata

from sentence_splitter import SentenceSplitter
from tqdm import tqdm

from garumzime.config import Config
from garumzime.constants import DATASETS_PATH, PROCESSED_DATA_PATH, TOKENIZER_PATH


def _remove_diacritics(input_str: str) -> str:
    nfkd_form = unicodedata.normalize("NFD", input_str)  # base + combining mark
    return "".join([c for c in nfkd_form if unicodedata.category(c) != "Mn"])


def _filter_text(text: str, tok: dict) -> str:
    text_nfc = unicodedata.normalize("NFC", text)
    result_text = ""
    for c in text_nfc:
        if c in tok["vocab"]:
            result_text += c
    return result_text


def _save_feature_target(text: str, out_f):
    feature = _remove_diacritics(text)
    target = text
    out = {"feature": feature, "target": target}
    out_f.write(json.dumps(out, ensure_ascii=False) + "\n")


def prepare_data(cfg: Config):
    dataset_path = f"{DATASETS_PATH / cfg.dataset_name}.jsonl"
    out_path = f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl"
    tokenizer_path = f"{TOKENIZER_PATH / cfg.tokenizer_name}.json"

    seq_len: int = cfg.seq_length

    sentence_splitter = SentenceSplitter(language="lv")

    with open(tokenizer_path) as f:
        tokenizer_dict = json.load(f)

    with open(dataset_path, "rb") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(list(in_f), desc="Processing documents..."):
            text = json.loads(line)["text"]
            filtered_text = _filter_text(text, tokenizer_dict)

            if not filtered_text:
                continue

            # split on sentences
            sentences = sentence_splitter.split(filtered_text)

            # packing sentences
            # TODO - i made the quick decision and just removed sentences which
            # are less than seq_len. this of course leads to data loss
            # TODO - find a way to include long sentences
            running_str = ""
            for s in sentences:
                if len(s) > seq_len:
                    continue

                if len(s) + len(running_str) > seq_len:
                    _save_feature_target(running_str, out_f)
                    running_str = s
                else:
                    running_str += s
            if running_str:
                _save_feature_target(running_str, out_f)


if __name__ == "__main__":
    cfg = Config.from_toml()
    prepare_data(cfg)
