# train_hnet_deepspeed.py
# Run with: deepspeed --num_gpus=1 train_hnet_deepspeed.py --max_bytes 20000000

import argparse
import os
import time
from functools import partial

import torch
import deepspeed
from torch.utils.data import IterableDataset, DataLoader

# Import H-Net modules (installed via pip -e hnet)
# The H-Net repo provides a wrapper that turns H-Net into a sequence model.
# We attempt to import plausible entry points; if import name differs, edit to match your hnet package.
try:
    from hnet.mixer_seq import HNetLanguageModel   # preferred if available
except Exception:
    # fallback: try other typical names (you may need to update this)
    from hnet.models.hnet import HNet as HNetCore
    # Minimal wrapper (if the repo has different API)
    class HNetLanguageModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = HNetCore(cfg)
            self.vocab_size = cfg.get("vocab_size", 256)
        def forward(self, input_ids):
            # expects input_ids: (B, L) bytes 0..255
            logits = self.model(input_ids)  # adjust if API differs
            return logits

import tensorflow_datasets as tfds

class WikiBytesDataset(IterableDataset):
    def __init__(self, seq_len=512, max_bytes=20_000_000, lang="en"):
        self.seq_len = seq_len
        self.max_bytes = max_bytes
        self.lang = lang

    def __iter__(self):
        ds = tfds.load("wiki40b", split="train", builder_kwargs={"lang": self.lang}, shuffle_files=True)
        buffer = bytearray()
        total = 0
        for example in tfds.as_numpy(ds):
            text = example["text"].decode("utf-8", errors="ignore")
            b = text.encode("utf-8", errors="ignore")
            buffer.extend(b)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len+1]
                inp = torch.tensor(list(chunk[:-1]), dtype=torch.long)
                tgt = torch.tensor(list(chunk[1:]), dtype=torch.long)
                buffer = buffer[self.seq_len:]
                total += self.seq_len
                yield {"input_ids": inp, "labels": tgt}
                if total >= self.max_bytes:
                    return

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}

def get_model_and_optimizer(cfg_path, device):
    import json
    cfg = json.load(open(cfg_path, "r"))
    # Build H-Net model from cfg (repo provides builder patterns; adjust if needed)
    # We assume HNetLanguageModel(cfg_dict) constructs the model.
    model = HNetLanguageModel(cfg)
    model = model.to(device)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_bytes", type=int, default=20_000_000)
    p.add_argument("--hnet_config", type=str, default="hnet/configs/hnet_tiny.json")
    p.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--accumulate", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = get_model_and_optimizer(args.hnet_config, device)

    # DeepSpeed init
    # Create optimizer placeholder; DeepSpeed will wrap it if configured in JSON.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, _, _ = deepspeed.initialize(
        args=argparse.Namespace(), model=model, optimizer=optimizer,
        config_params=None,  # we'll pass config file below
        dist_init_required=False,
    )

    # load deepspeed config explicitly using engine.init args (workaround)
    # NOTE: deepspeed.initialize normally reads cmdline for --deepspeed_config; if needed, set env var
    # Simpler: run the script through the `deepspeed` launcher with --deepspeed_config
    # We'll proceed assuming user runs via: deepspeed --num_gpus=1 --deepspeed_config deepspeed_config.json train_hnet_deepspeed.py

    dataset = WikiBytesDataset(seq_len=args.seq_len, max_bytes=args.max_bytes)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    step = 0
    epoch_loss = 0.0
    start_time = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)  # (B, L, vocab)
        # H-Net's output shape may vary; adapt as needed.
        # assume logits shape (B, L, V)
        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(logits.view(-1, logits.size(-1)), labels.view(-1))
        model.backward(loss)
        model.step()

        epoch_loss += loss.item()
        step += 1
        if step % args.print_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | loss {epoch_loss / args.print_every:.4f} | elapsed {elapsed:.1f}s")
            epoch_loss = 0.0
        if step >= args.steps:
            break

    print("Training finished. Steps:", step)

if __name__ == "__main__":
    main()
