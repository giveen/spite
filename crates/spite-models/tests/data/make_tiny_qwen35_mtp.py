#!/usr/bin/env python3
"""Derive the MTP fixtures from tiny-qwen35-f16.gguf by appending a NextN block.

    python3 make_tiny_qwen35_mtp.py   (run from this directory)

Writes two files next to the source model; both keep its trunk byte-for-byte
and add `blk.2` (full attention, dense FFN) plus metadata block_count=3,
nextn_predict_layers=1:

  tiny-qwen35-mtp-identity-f16.gguf
      eh_proj = [0 | I] picks the hidden half of the stem, attn_output and
      ffn_down are zero and every norm weight is 1, so the head's logits are
      exactly a positive multiple of the trunk's logits for the same h_t.
  tiny-qwen35-mtp-full-f16.gguf
      eh_proj is a fixed pseudo-random matrix and the block copies blk.1, so
      the draft depends on the token, the hidden state and its KV history.
"""

import struct
import sys
from pathlib import Path

SRC = "tiny-qwen35-f16.gguf"
F32, F16 = 0, 1
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, data):
        self.d, self.o = data, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0]

    def string(self):
        n = self.take("Q")
        s = self.d[self.o:self.o + n].decode()
        self.o += n
        return s

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            return (et, [self.value(et) for _ in range(n)])
        return self.take(SCALAR[t])


def enc_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def enc_value(t, v):
    if t == 8:
        return enc_str(v)
    if t == 9:
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(enc_value(et, x) for x in items)
    return struct.pack("<" + SCALAR[t], v)


def align(n, a):
    return (n + a - 1) // a * a


def read_gguf(path):
    r = Reader(Path(path).read_bytes())
    assert r.d[:4] == b"GGUF"
    r.o = 4
    version, n_t, n_kv = r.take("I"), r.take("Q"), r.take("Q")
    kv = []
    for _ in range(n_kv):
        k, t = r.string(), r.take("I")
        kv.append([k, t, r.value(t)])
    infos = []
    for _ in range(n_t):
        name, nd = r.string(), r.take("I")
        ne = [r.take("Q") for _ in range(nd)]
        infos.append((name, ne, r.take("I"), r.take("Q")))
    alignment = next((v for k, _, v in kv if k == "general.alignment"), 32)
    base = align(r.o, alignment)
    tensors = []
    for name, ne, ty, off in infos:
        n = 1
        for x in ne:
            n *= x
        size = n * (4 if ty == F32 else 2)
        tensors.append((name, ne, ty, r.d[base + off:base + off + size]))
    return version, kv, tensors, alignment


def write_gguf(path, version, kv, tensors, alignment):
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kv))
    head += b"".join(enc_str(k) + struct.pack("<I", t) + enc_value(t, v) for k, t, v in kv)
    infos, blob = b"", b""
    for name, ne, ty, data in tensors:
        off = align(len(blob), alignment)
        blob += b"\0" * (off - len(blob)) + data
        infos += enc_str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", ty, off)
    out = head + infos
    out += b"\0" * (align(len(out), alignment) - len(out)) + blob
    Path(path).write_bytes(out)


def f32(vals):
    return struct.pack(f"<{len(vals)}f", *vals)


def lcg(seed):
    s = seed
    while True:
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield ((s >> 8) / float(1 << 24) - 0.5) * 0.2


def build(variant):
    version, kv, tensors, alignment = read_gguf(SRC)
    by_name = {t[0]: t for t in tensors}
    meta = {k: (t, v) for k, t, v in kv}
    d = meta["qwen35.embedding_length"][1]
    for entry in kv:
        k = entry[0]
        if k == "qwen35.block_count":
            entry[2] += 1
        elif k == "qwen35.nextn_predict_layers":
            entry[2] = 1
        elif k == "qwen35.attention.recurrent_layers":
            et, items = entry[2]
            entry[2] = (et, items + [type(items[-1])(0)])  # NextN block: full attention

    b = "blk.2"
    ones = f32([1.0] * d)
    new = [
        (f"{b}.nextn.enorm.weight", [d], F32, ones),
        (f"{b}.nextn.hnorm.weight", [d], F32, ones),
        (f"{b}.nextn.shared_head_norm.weight", [d], F32, ones),
        (f"{b}.attn_norm.weight", [d], F32, ones),
        (f"{b}.post_attention_norm.weight", [d], F32, ones),
    ]
    if variant == "identity":
        eh = [0.0] * (2 * d * d)
        for r in range(d):
            eh[r * 2 * d + d + r] = 1.0
    else:
        g = lcg(7)
        eh = [next(g) for _ in range(2 * d * d)]
    new.append((f"{b}.nextn.eh_proj.weight", [2 * d, d], F32, f32(eh)))
    for suffix in ["attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
                   "ffn_gate", "ffn_up", "attn_output", "ffn_down"]:
        _, ne, ty, data = by_name[f"blk.1.{suffix}.weight"]
        if variant == "identity" and suffix in ("attn_output", "ffn_down"):
            data = b"\0" * len(data)
        new.append((f"{b}.{suffix}.weight", ne, ty, data))
    write_gguf(f"tiny-qwen35-mtp-{variant}-f16.gguf", version, kv, tensors + new, alignment)


if __name__ == "__main__":
    if not Path(SRC).exists():
        sys.exit(f"run from the directory containing {SRC}")
    for v in ("identity", "full"):
        build(v)
