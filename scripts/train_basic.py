import torch
from torch.optim import Adam
from x_transformers import XTransformer

from garumzime.config import Config
from garumzime.loader import Loader
from garumzime.prepare_data import prepare_data

LEARNING_RATE = 3e-4


def train():
    cfg = Config.from_toml()
    prepare_data(cfg)

    loader = Loader(cfg)

    for f, t in loader:
        break

    model = XTransformer(
        dim=512,
        enc_num_tokens=256,
        enc_depth=6,
        enc_heads=8,
        enc_max_seq_len=128,
        dec_num_tokens=256,
        dec_depth=6,
        dec_heads=8,
        dec_max_seq_len=128,
        tie_token_emb=True,  # tie embeddings of encoder and decoder
    )

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    src = torch.randint(0, 256, (1, 128))
    src_mask = torch.ones_like(src).bool()
    tgt = torch.randint(0, 256, (1, 128))
    for i in range(100):
        loss = model(src, tgt, mask=src_mask)  # (1, 1024, 512)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch: {i}, loss: {loss:.8f}")


if __name__ == "__main__":
    train()
