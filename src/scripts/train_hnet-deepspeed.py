# train_unified_deepspeed.py
# Usage examples:
#   deepspeed scripts/train_unified_deepspeed.py --model gptneox --deepspeed_config configs/deepseek_config.json
#   deepspeed scripts/train_unified_deepspeed.py --model hnet --hnet_config configs/hnet-tiny_config.json --deepspeed_config configs/deepseek_config.json
# Notes:
# - Uses Hugging Face Trainer to integrate DeepSpeed for both GPTNeoX and HNet.
# - For HNet, we wrap the model to compute loss when labels are provided.

import os
import argparse
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys

import torch
from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds

from transformers import Trainer, TrainingArguments
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

# --- HNet import ---
try:
    from hnet.models.mixer_seq import HNetForCausalLM
    from hnet.models.config_hnet import HNetConfig
    from hnet.models.hnet import HNet
except Exception as e:
    HNetForCausalLM = None
    HNetConfig = None
    HNet = None

# -----------------
# Dataset (byte-level Wiki40B streaming, same as existing project style)
# -----------------
class WikiBytesDataset(IterableDataset):
    def __init__(self, seq_len=512, max_bytes=20_000_000, lang="en", split="train"):
        self.seq_len = seq_len
        self.max_bytes = max_bytes
        self.lang = lang
        self.split = split

    def __iter__(self):
        ds = tfds.load("wiki40b", split=self.split, builder_kwargs={"lang": self.lang}, shuffle_files=True)
        buffer = bytearray()
        total = 0
        for example in tfds.as_numpy(ds):
            text = example["text"].decode("utf-8", errors="ignore")
            b = text.encode("utf-8", errors="ignore")
            buffer.extend(b)
            # emit overlapping chunks packed as next-token prediction
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                inp = torch.tensor(list(chunk[:-1]), dtype=torch.long)
                tgt = torch.tensor(list(chunk[1:]), dtype=torch.long)
                # Shift buffer by seq_len (packed contiguous without gap)
                buffer = buffer[self.seq_len :]
                total += self.seq_len
                yield {"input_ids": inp, "labels": tgt}
                if total >= self.max_bytes:
                    return

def data_collator(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}

# -----------------
# Loss wrapper so Trainer works with HNet (and plain Torch modules generally)
# -----------------
class LossComputingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, vocab_size: int = 256):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size

    def forward(self, input_ids=None, labels=None, **kwargs):
        # Expect model to return logits of shape (B, L, V)
        outputs = self.model(input_ids=input_ids, **kwargs) if hasattr(self.model, "__call__") else self.model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = None
        if labels is not None:
            # Standard next-token cross-entropy
            loss_f = torch.nn.CrossEntropyLoss()
            loss = loss_f(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

# -----------------
# Custom Trainer to read 'loss' from dict-returning models
# -----------------
class DictReturningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

def build_hnet_model(config_path: str):
    assert HNetForCausalLM is not None, "HNet is not available in this environment"
    with open(config_path, "r") as f:
        cfg_json = json.load(f)
    cfg = HNetConfig(**cfg_json)
    model = HNetForCausalLM(cfg)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hnet_config", type=str, default="configs/hnet-tiny_config.json")
    ap.add_argument("--deepspeed_config", type=str, default="configs/deepseek_config.json")
    ap.add_argument("--output_dir", type=str, default="checkpoints")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--max_bytes", type=int, default=20_000_000)
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--train_micro_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", default=True, help="Enable fp16 mixed precision")
    ap.add_argument("--logging_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--eval_bytes", type=int, default=512 * 500)

    # Accept common DeepSpeed/launcher-injected args without crashing:
    ap.add_argument("--local_rank", type=int, default=0, help="Injected by launcher")
    ap.add_argument("--steps", type=int, default=None, help="Optional: steps from wrapper")
    ap.add_argument("--print_every", type=int, default=None, help="Optional: logging freq from wrapper")
    ap.add_argument("--save_dir", type=str, default=None, help="Optional: override output dir from wrapper")

    # parse_known_args prevents unexpected launcher flags from crashing the script
    args, unknown = ap.parse_known_args()
    if unknown:
        print("Warning: ignoring unknown launcher args:", unknown, file=sys.stderr)

    # allow wrapper-provided save_dir to override output_dir
    if args.save_dir:
        args.output_dir = args.save_dir

    # normalize boolean
    args.fp16 = bool(args.fp16)


    # --- Create output directory ---
    # Read hnet_config to decide the output directory name
    with open(args.hnet_config, 'r') as f:
        hnet_config_json = json.load(f)
    
    use_gptneox_inside_hnet = hnet_config_json.get("use_gptneox_backbone", False)
    
    output_dir_name = "hnet_with_gptneox" if use_gptneox_inside_hnet else "hnet"
    output_dir = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build datasets
    train_dataset = WikiBytesDataset(seq_len=args.seq_len, max_bytes=args.max_bytes, lang=args.lang, split="train")
    eval_dataset = WikiBytesDataset(seq_len=args.seq_len, max_bytes=args.eval_bytes, lang=args.lang, split="train")  # quick eval

    vocab_size = 256  # byte-level

    # --- Build HNet model and its Trainer ---
    print("--- Setting up HNet ---")
    hnet_model = build_hnet_model(args.hnet_config)
    hnet_model = LossComputingWrapper(hnet_model, vocab_size=vocab_size)

    hnet_training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
        report_to=[],
    )

    hnet_trainer = DictReturningTrainer(
        model=hnet_model,
        args=hnet_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # --- Train model ---
    print("\n--- Starting HNet training ---")
    hnet_trainer.train()
    print("\n--- HNet training finished ---")


if __name__ == "__main__":
    main()
