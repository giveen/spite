#!/usr/bin/env python3
"""Compare two raw f32 logit dumps [n_pos, vocab]: per-position max |diff|, top-1 agreement,
and top-5 overlap. Exit 1 if any position's top-1 differs or max |diff| > --tol.

  compare_logits.py ref.f32 test.f32 VOCAB [--tol 1e-2]
"""
import array, sys

def load(p):
    a = array.array("f"); a.frombytes(open(p, "rb").read()); return a

def main():
    if len(sys.argv) < 4: sys.exit(__doc__)
    ref, tst, vocab = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-2
    if len(ref) != len(tst) or len(ref) % vocab:
        sys.exit(f"shape mismatch: {len(ref)} vs {len(tst)} floats, vocab {vocab}")
    bad = False
    for i in range(len(ref) // vocab):
        r, t = ref[i*vocab:(i+1)*vocab], tst[i*vocab:(i+1)*vocab]
        d = max(abs(x - y) for x, y in zip(r, t))
        top = lambda v: sorted(range(vocab), key=lambda k: -v[k])[:5]
        tr, tt = top(r), top(t)
        ok = tr[0] == tt[0] and d <= tol
        bad |= not ok
        print(f"pos {i:3d}: max|diff|={d:.3e} top1 ref={tr[0]} test={tt[0]} top5-overlap={len(set(tr)&set(tt))}/5 {'OK' if ok else 'FAIL'}")
    sys.exit(1 if bad else 0)

main()
