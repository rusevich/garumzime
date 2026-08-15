import tomllib
from dataclasses import dataclass

from garumzime.constants import CONFIG_PATH


@dataclass(frozen=True)
class Config:
    dataset_name: str
    seq_length: int
    tokenizer_name: str

    @staticmethod
    def from_toml():
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        dataset_name = data["dataset_name"]
        seq_length = data["seq_length"]
        tokenizer_name = data["tokenizer_name"]

        return Config(dataset_name, seq_length, tokenizer_name)


if __name__ == "__main__":
    cfg = Config.from_toml()
    print(cfg)
