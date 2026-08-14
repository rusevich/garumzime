import tomllib
from dataclasses import dataclass

from garumzime.constants import CONFIG_PATH


@dataclass(frozen=True)
class Config:
    dataset_name: str
    seq_length: int

    @staticmethod
    def from_toml():
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        dataset_name = data["dataset_name"]
        seq_length = data["seq_length"]

        return Config(dataset_name, seq_length)


if __name__ == "__main__":
    cfg = Config.from_toml()
    print(cfg)
