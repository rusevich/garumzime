import argparse
import json

import torch
from x_transformers import XTransformer

from garumzime.config import Config
from garumzime.constants import LOCAL_DATA, PROCESSED_DATA_PATH, TOKENIZER_PATH

MODEL_PATH = LOCAL_DATA / "models" / "basic_model.pth"

# edit me. same shape as the jsonl. "target" optional; without it pass --start <char>
EXAMPLES = [
    {
        "feature": "Sodien ir loti skaista diena, un saule spid debesis.",
        "target": "Šodien ir ļoti skaista diena, un saule spīd debesīs.",
    },
    {
        "feature": "Es macos latviesu valodu jau piecus gadus.",
        "target": "Es mācos latviešu valodu jau piecus gadus.",
    },
    {
        "feature": "Riga list lietus, un ielas ir slapjas.",
        "target": "Rīgā līst lietus, un ielas ir slapjas.",
    },
    {
        "feature": "Cels uz mezu ir gars un likumots.",
        "target": "Ceļš uz mežu ir garš un līkumots.",
    },
    {
        "feature": "2:1 Un tresaja diena Kana, Galileja, bija kazas.",
        "target": "2:1 Un trešajā dienā Kānā, Galilejā, bija kāzas.",
    },
]


def load_tok(cfg):
    with open(f"{TOKENIZER_PATH / cfg.tokenizer_name}.json") as f:
        vocab = json.load(f)["vocab"]
    tok2id = {k: int(v["id"]) for k, v in vocab.items()}
    return tok2id, {v: k for k, v in tok2id.items()}


def to_tokens(x, split):
    if isinstance(x, str):
        return x.split(split) if split else list(x)
    return x


def encode(tokens, tok2id, unk_id, pad_id, seq_len):
    ids = [tok2id.get(t, unk_id) for t in tokens][:seq_len]
    return ids + [pad_id] * (seq_len - len(ids))


def decode(ids, id2tok, pad_id, sep):
    return sep.join(id2tok.get(int(i), "<unk>") for i in ids if int(i) != pad_id)


def read_samples(source, cfg, split):
    """yields (feature_tokens, target_tokens|None)"""
    if source == "examples":
        rows = EXAMPLES
    else:
        path = f"{PROCESSED_DATA_PATH / cfg.dataset_name}.jsonl" if source == "train" else source
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line) if line.startswith("{") else {"feature": line})
    for d in rows:
        tgt = d.get("target")
        yield to_tokens(d["feature"], split), to_tokens(tgt, split) if tgt else None


@torch.no_grad()
def greedy(model, src, mask, seed, steps):
    enc = model.encoder(src, mask=mask, return_embeddings=True)
    out = seed
    for _ in range(steps):
        logits = model.decoder.net(out, context=enc, context_mask=mask)[:, -1]
        out = torch.cat([out, logits.argmax(-1, keepdim=True)], dim=-1)
    return out[0]


@torch.no_grad()
def teacher_forced(model, src, mask, tgt):
    enc = model.encoder(src, mask=mask, return_embeddings=True)
    logits = model.decoder.net(tgt[:, :-1], context=enc, context_mask=mask)
    return torch.cat([tgt[0, :1], logits[0].argmax(-1)])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="examples", help="'examples', 'train', or a file path")
    p.add_argument("-n", "--num-samples", type=int, default=5)
    p.add_argument("--sep", default="", help="join separator when printing")
    p.add_argument("--start", default=None, help="decoder seed token, default: target[0]")
    p.add_argument("--split", default=None, help="split strings on this instead of per-char")
    p.add_argument("--full", action="store_true", help="generate full seq_length, don't trim")
    p.add_argument("--teacher", action="store_true", help="teacher-forced argmax, needs target")
    a = p.parse_args()

    cfg = Config.from_toml()
    tok2id, id2tok = load_tok(cfg)
    unk_id, pad_id = tok2id["<unk>"], tok2id["<pad>"]

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
        tie_token_emb=True,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    for i, (feat_toks, tgt_toks) in enumerate(read_samples(a.data, cfg, a.split)):
        if i >= a.num_samples:
            break

        src = torch.tensor([encode(feat_toks, tok2id, unk_id, pad_id, cfg.seq_length)])
        mask = src != pad_id

        if a.teacher:
            if not tgt_toks:
                raise SystemExit(f"sample {i} has no target, --teacher needs one")
            tgt = torch.tensor([encode(tgt_toks, tok2id, unk_id, pad_id, cfg.seq_length)])
            pred = teacher_forced(model, src, mask, tgt)
        else:
            if a.start:
                start_id = tok2id.get(a.start, unk_id)
            elif tgt_toks:
                start_id = tok2id.get(tgt_toks[0], unk_id)
            else:
                raise SystemExit(f"sample {i} has no target, pass --start <token>")
            steps = cfg.seq_length - 1 if a.full else min(len(feat_toks) - 1, cfg.seq_length - 1)
            pred = greedy(model, src, mask, torch.tensor([[start_id]], dtype=torch.long), steps)

        if not a.full:
            pred = pred[: len(feat_toks)]

        print(f"[{i}] src : {decode(src[0], id2tok, pad_id, a.sep)}")
        if tgt_toks:
            print(f"[{i}] tgt : {a.sep.join(tgt_toks)}")
        print(f"[{i}] pred: {decode(pred, id2tok, pad_id, a.sep)}")
        print()


if __name__ == "__main__":
    main()
