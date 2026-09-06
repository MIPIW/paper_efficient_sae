"""Emit a style-labelled copy of the Amazon-reviews splits for the content/style pilot.

The trainer supervises branch P from `sentiment_label` and branch Q from `topic_label`. To run the
pilot with Q carrying *style* instead of topic, nothing in the trainer needs to change -- only the
dataset does: this writes an otherwise byte-identical copy of each split in which `topic_label`
holds a binary lexical-diversity label. The original topic string/label is preserved under
`orig_topic_label` so nothing is destroyed.

Style = lexical diversity, measured as MATTR (moving-average type-token ratio): the mean over all
sliding windows of `--window` words of |unique words in window| / window.

Getting this measure right took three attempts and the failures are instructive:

1. Plain TTR = |unique| / |total| is dominated by length -- corr(TTR, length) = -0.53 on this corpus.
   A global median split would produce a *length* label wearing a style label's clothes.
2. Binning into length deciles and splitting at each bin's own median fixes the correlation in
   principle but not in practice: 26.5% of documents have TTR exactly 1.0 (every word unique), and
   the three shortest deciles have median TTR = 1.0 with 53-78% of their mass tied there. No document
   can exceed 1.0, so every short document was forced into one class -- yielding a 0.333 positive
   rate and a residual +0.11 length correlation.
3. A fixed 20-word window removes the length dependence but has granularity 1/20, so tercile
   boundaries collapse onto tie values and no margin exists.

MATTR averages many overlapping windows, so it is near-continuous (few ties) while remaining
length-robust. Documents shorter than `--min_words` are dropped -- lexical diversity is not
measurable on them, and forcing a label would be inventing signal. The label is then a TERCILE split
with the middle third DISCARDED, which gives a real margin between classes instead of a tie-riddled
boundary.

The script prints label/length and label/sentiment correlations as validity checks -- if
label/sentiment is materially non-zero the two branches would be receiving correlated targets and
the pilot would be void.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("train.jsonl", "eval.jsonl", "withheld_rarest_cell.jsonl")


def mattr_and_length(text: str, window: int, min_words: int, cap: int = 200):
    """MATTR over sliding windows, via a rolling count so cost is O(n) not O(n*window).

    `cap` bounds the words considered on very long reviews; MATTR is already an average over
    windows, so truncating a long tail changes it only marginally and keeps the pass over 2.9M
    documents tractable.
    """
    words = text.lower().split()
    n_total = len(words)
    if n_total < min_words:
        return None, n_total
    words = words[:cap]
    n = len(words)
    counts: dict[str, int] = {}
    for w in words[:window]:
        counts[w] = counts.get(w, 0) + 1
    acc, n_windows = len(counts), 1
    for i in range(window, n):
        out_w = words[i - window]
        counts[out_w] -= 1
        if counts[out_w] == 0:
            del counts[out_w]
        in_w = words[i]
        counts[in_w] = counts.get(in_w, 0) + 1
        acc += len(counts)
        n_windows += 1
    return acc / (n_windows * window), n_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(ROOT / "data" / "amazon_reviews_full500k"))
    ap.add_argument("--dst", type=str, default=str(ROOT / "data" / "amazon_reviews_style500k"))
    ap.add_argument("--window", type=int, default=15, help="MATTR window in words.")
    ap.add_argument("--min_words", type=int, default=25, help="Documents shorter than this are dropped.")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src_path = src / split
        if not src_path.exists():
            print(f"[{split}] missing in source, skipping", flush=True)
            continue

        # Pass 1: MATTR + length for every row, to fix the tercile cut points exactly.
        vals, lens = [], []
        with src_path.open(encoding="utf-8") as f:
            for line in f:
                v, n = mattr_and_length(json.loads(line)["text"], args.window, args.min_words)
                vals.append(np.nan if v is None else v)
                lens.append(n)
        vals_a = np.asarray(vals, dtype=np.float64)
        lens_a = np.asarray(lens, dtype=np.int64)
        n_rows = len(vals_a)
        eligible = ~np.isnan(vals_a)
        lo, hi = np.quantile(vals_a[eligible], [1 / 3, 2 / 3])

        # -1 marks "drop": too short to measure, or in the discarded middle tercile.
        labels = np.full(n_rows, -1, dtype=np.int64)
        labels[eligible & (vals_a <= lo)] = 0
        labels[eligible & (vals_a >= hi)] = 1

        # Pass 2: rewrite, keeping only labelled rows.
        kept_labels, kept_lens, kept_sent = [], [], []
        with src_path.open(encoding="utf-8") as fin, (dst / split).open("w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if labels[i] < 0:
                    continue
                row = json.loads(line)
                row["orig_topic_label"] = int(row["topic_label"])
                row["orig_topic"] = row.get("topic")
                row["mattr"] = float(vals_a[i])
                row["topic_label"] = int(labels[i])
                row["topic"] = "high_diversity" if labels[i] else "low_diversity"
                fout.write(json.dumps(row) + "\n")
                kept_labels.append(labels[i])
                kept_lens.append(lens_a[i])
                kept_sent.append(int(row["sentiment_label"]))

        lab = np.asarray(kept_labels, dtype=np.float64)
        ln = np.asarray(kept_lens, dtype=np.float64)
        sn = np.asarray(kept_sent, dtype=np.float64)
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0  # noqa: E731
        print(f"[{split}] in={n_rows} kept={len(lab)} ({len(lab)/n_rows:.3f}) "
              f"cut=({lo:.4f},{hi:.4f}) pos_rate={lab.mean():.4f} "
              f"corr(label,length)={corr(lab, ln):+.4f} "
              f"corr(label,sentiment)={corr(lab, sn):+.4f} "
              f"mean_mattr c0={vals_a[labels == 0].mean():.4f} c1={vals_a[labels == 1].mean():.4f}",
              flush=True)

    meta_src = src / "metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta["orig_topic_label_map"] = meta.get("topic_label_map")
        meta["topic_label_map"] = {"low_diversity": 0, "high_diversity": 1}
        meta["style_label_note"] = (
            f"topic_label slot carries a binary lexical-diversity label: MATTR(window={args.window}) "
            f"on documents of >={args.min_words} words, tercile split with the middle third discarded"
        )
        (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    for extra in src.glob("*.json"):
        if extra.name != "metadata.json" and not (dst / extra.name).exists():
            shutil.copy(extra, dst / extra.name)
    print(f"Wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
