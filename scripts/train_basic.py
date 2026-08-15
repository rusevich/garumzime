import torch
from torch.optim import Adam
from x_transformers import XTransformer

from garumzime.config import Config
from garumzime.constants import LOCAL_DATA
from garumzime.loader import Loader

torch.manual_seed(0)
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
ITERATIONS = 1000


def get_batch(loader, batch_idx, seq_len):
    start_idx, end_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
    batch_feature = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.long)
    batch_target = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.long)
    batch_mask = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)
    for i in range(start_idx, end_idx):
        feature, target = loader[i]
        f = torch.as_tensor(feature, dtype=torch.long)
        t = torch.as_tensor(target, dtype=torch.long)
        m = f != 0

        batch_feature[i % BATCH_SIZE, :] = f
        batch_target[i % BATCH_SIZE, :] = t
        batch_mask[i % BATCH_SIZE, :] = m

    return batch_feature, batch_target, batch_mask


def train():
    cfg = Config.from_toml()
    loader = Loader(cfg)
    model_save_path = LOCAL_DATA / "models" / "basic_model.pth"

    model = XTransformer(
        dim=512,
        enc_num_tokens=120,
        enc_depth=3,
        enc_heads=8,
        enc_max_seq_len=cfg.seq_length,
        dec_num_tokens=120,
        dec_depth=3,
        dec_heads=8,
        dec_max_seq_len=cfg.seq_length,
        tie_token_emb=True,  # tie embeddings of encoder and decoder
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    try:
        for iteration in range(ITERATIONS):
            features, targets, masks = get_batch(loader, iteration, cfg.seq_length)
            loss = model(features, targets, mask=masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"iteration: {iteration}, loss: {loss:.8f}")
    except Exception as e:
        print("i'm not sorry for being lazy, i don't fucking care")
        print(e)
    print("saving the model...")
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train()
