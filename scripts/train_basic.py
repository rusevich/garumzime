import torch
from garumzime.loader import DiacriticDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from x_transformers import XTransformer

from garumzime.config import Config
from garumzime.constants import LOCAL_DATA

torch.manual_seed(0)
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
ITERATIONS = 1000
CHECKPOINT_RATE = 200

DEVICE = "cpu"
if torch.cuda_is_available():
    DEVICE = "cuda"

print(f"Device used: {DEVICE}")


def train():
    cfg = Config.from_toml()
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
        ignore_index=0,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    ds = DiacriticDataset(cfg)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=ds.collate, drop_last=True
    )

    try:
        for iteration, (feature, target, mask) in enumerate(loader):
            feature, target, mask = feature.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)
            loss = model(feature, target, mask=mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if iteration >= ITERATIONS:
                break
            print(f"iteration: {iteration}, loss: {loss:.8f}")

            if iteration % CHECKPOINT_RATE == 0:
                print("saving the model...")
                torch.save(model.state_dict(), model_save_path)

    except (Exception, KeyboardInterrupt) as e:
        print(e)
    print("saving the model...")
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    train()
