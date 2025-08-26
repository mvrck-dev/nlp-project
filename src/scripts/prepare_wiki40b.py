# prepare_wiki40b.py
# Usage example: python prepare_wiki40b.py --max_bytes 20000000 --seq_len 512 --preview 1

import argparse
import tensorflow_datasets as tfds

def stream_bytes(lang="en", split="train", max_bytes=20_000_000, seq_len=512, preview=0):
    """
    Stream Wiki40B text, emit byte sequences of length seq_len.
    Yields (input_bytes, target_bytes) where both are lists/arrays of ints 0..255
    """
    ds = tfds.load("wiki40b", split=split, builder_kwargs={"lang": lang}, shuffle_files=True)
    total = 0
    buffer = bytearray()
    for example in tfds.as_numpy(ds):
        text = example["text"].decode("utf-8", errors="ignore")
        # convert to bytes (UTF-8)
        b = text.encode("utf-8", errors="ignore")
        buffer.extend(b)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len+1]
            input_bytes = list(chunk[:-1])
            target_bytes = list(chunk[1:])
            yield input_bytes, target_bytes
            buffer = buffer[seq_len:]  # sliding window (non-overlapping)
            total += seq_len
            if total >= max_bytes:
                return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_bytes", type=int, default=20_000_000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--preview", type=int, default=0)
    args = p.parse_args()

    i = 0
    for inp, tgt in stream_bytes(lang=args.lang, max_bytes=args.max_bytes, seq_len=args.seq_len):
        if args.preview:
            print("sample input bytes:", inp[:32])
            print("as text:", bytes(inp).decode("utf-8", errors="ignore")[:120])
            break
        i += 1
    print("Done streaming. sequences:", i)
