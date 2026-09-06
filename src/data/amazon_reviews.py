"""Amazon Reviews 2023 data preparation for KronSAE compositional experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import fsspec
import torch
from huggingface_hub import HfApi
from torch.utils.data import Dataset, Sampler

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

# Prefer smaller/medium categories to keep preprocessing tractable while still realistic.
PREFERRED_CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Office_Products",
    "Pet_Supplies",
    "Electronics",
    "Home_and_Kitchen",
]

NEGATIVE = 0
POSITIVE = 1
SENTIMENT_NAME = {NEGATIVE: "negative", POSITIVE: "positive"}


@dataclass
class PreparedPaths:
    root: Path
    train_jsonl: Path
    eval_jsonl: Path
    withheld_jsonl: Path
    metadata_json: Path


def prepared_paths(output_dir: str | Path) -> PreparedPaths:
    root = Path(output_dir)
    return PreparedPaths(
        root=root,
        train_jsonl=root / "train.jsonl",
        eval_jsonl=root / "eval.jsonl",
        withheld_jsonl=root / "withheld_rarest_cell.jsonl",
        metadata_json=root / "metadata.json",
    )


def _build_category_map_from_repo() -> dict[str, str]:
    api = HfApi()
    files = api.list_repo_files(DATASET_NAME, repo_type="dataset")

    mapping: dict[str, str] = {}
    prefix = "raw/review_categories/"
    for filename in files:
        if not filename.startswith(prefix) or not filename.endswith(".jsonl"):
            continue
        category = Path(filename).stem
        mapping[category] = filename

    if not mapping:
        raise RuntimeError("No raw/review_categories/*.jsonl files found in dataset repo")

    return mapping


def resolve_categories(
    available_category_to_path: dict[str, str],
    categories: list[str] | None,
    num_categories: int,
) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    seen = set()

    requested = categories if categories is not None else PREFERRED_CATEGORIES

    for cat in requested:
        if cat in available_category_to_path and cat not in seen:
            resolved.append((cat, available_category_to_path[cat]))
            seen.add(cat)
        if len(resolved) >= num_categories:
            break

    if len(resolved) < num_categories:
        for cat in sorted(available_category_to_path.keys()):
            if cat in seen:
                continue
            resolved.append((cat, available_category_to_path[cat]))
            seen.add(cat)
            if len(resolved) >= num_categories:
                break

    if len(resolved) < num_categories:
        raise RuntimeError(f"Could only resolve {len(resolved)} categories, need {num_categories}")

    return resolved[:num_categories]


def _extract_rating(row: dict[str, Any]) -> float | None:
    for key in ("rating", "stars", "overall", "score"):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_text(row: dict[str, Any]) -> str | None:
    candidates: list[str] = []

    for key in ("text", "review_body", "content", "reviewText", "body"):
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                candidates.append(value)

    title = row.get("title")
    if isinstance(title, str):
        title = title.strip()
        if title:
            if candidates:
                return f"{title}\n\n{candidates[0]}"
            return title

    if candidates:
        return candidates[0]
    return None


def _binarize_sentiment(rating: float) -> int | None:
    if rating <= 2.0:
        return NEGATIVE
    if rating >= 4.0:
        return POSITIVE
    return None


def collect_category_samples(
    category_name: str,
    category_filename: str,
    topic_label: int,
    sample_per_category: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    remote_path = f"hf://datasets/{DATASET_NAME}/{category_filename}"

    # Stream JSONL directly from Hub so we can stop early after sampling.
    with fsspec.open(remote_path, "r") as f:
        for line in f:
            row = json.loads(line)

            rating = _extract_rating(row)
            if rating is None:
                continue

            sentiment = _binarize_sentiment(rating)
            if sentiment is None:
                continue

            text = _extract_text(row)
            if text is None:
                continue

            records.append(
                {
                    "text": text,
                    "topic": category_name,
                    "topic_label": topic_label,
                    "sentiment_label": sentiment,
                    "rating": float(rating),
                }
            )

            if len(records) >= sample_per_category:
                break

    if not records:
        raise RuntimeError(f"No usable records collected for category={category_name}")

    return records


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def prepare_amazon_reviews(
    output_dir: str | Path,
    categories: list[str] | None = None,
    num_categories: int = 6,
    sample_per_category: int = 50_000,
    eval_fraction: float = 0.02,
    seed: int = 42,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Prepare train/eval/withheld splits and write empirical count metadata.

    Rarest non-empty (category, sentiment) cell is fully withheld from training.
    """

    paths = prepared_paths(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    if (
        not force_rebuild
        and paths.train_jsonl.exists()
        and paths.eval_jsonl.exists()
        and paths.withheld_jsonl.exists()
        and paths.metadata_json.exists()
    ):
        with paths.metadata_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    available_map = _build_category_map_from_repo()
    selected = resolve_categories(
        available_category_to_path=available_map,
        categories=categories,
        num_categories=num_categories,
    )

    all_records: list[dict[str, Any]] = []
    category_topic_to_label: dict[str, int] = {}

    for topic_label, (category_name, category_filename) in enumerate(selected):
        category_topic_to_label[category_name] = topic_label
        category_records = collect_category_samples(
            category_name=category_name,
            category_filename=category_filename,
            topic_label=topic_label,
            sample_per_category=sample_per_category,
        )
        all_records.extend(category_records)

    counts = defaultdict(lambda: {NEGATIVE: 0, POSITIVE: 0})
    per_cell_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in all_records:
        category = row["topic"]
        sentiment = int(row["sentiment_label"])
        counts[category][sentiment] += 1
        per_cell_records[(category, sentiment)].append(row)

    candidate_cells: list[tuple[str, int, int]] = []
    for category in sorted(counts.keys()):
        for sentiment in (NEGATIVE, POSITIVE):
            count = counts[category][sentiment]
            if count > 0:
                candidate_cells.append((category, sentiment, count))

    if not candidate_cells:
        raise RuntimeError("No non-empty (category, sentiment) cells found")

    rarest_category, rarest_sentiment, rarest_count = min(
        candidate_cells,
        key=lambda x: (x[2], x[0], x[1]),
    )

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    withheld_rows: list[dict[str, Any]] = []

    split_counts = defaultdict(lambda: {"train": 0, "eval": 0, "withheld": 0})

    for (category, sentiment), rows in per_cell_records.items():
        rows = rows.copy()
        rng.shuffle(rows)

        if category == rarest_category and sentiment == rarest_sentiment:
            withheld_rows.extend(rows)
            split_counts[(category, sentiment)]["withheld"] += len(rows)
            continue

        if len(rows) < 2:
            n_eval = 0
        else:
            n_eval = max(1, int(round(len(rows) * eval_fraction)))
            n_eval = min(n_eval, len(rows) - 1)

        eval_part = rows[:n_eval]
        train_part = rows[n_eval:]

        eval_rows.extend(eval_part)
        train_rows.extend(train_part)

        split_counts[(category, sentiment)]["eval"] += len(eval_part)
        split_counts[(category, sentiment)]["train"] += len(train_part)

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    _write_jsonl(paths.train_jsonl, train_rows)
    _write_jsonl(paths.eval_jsonl, eval_rows)
    _write_jsonl(paths.withheld_jsonl, withheld_rows)

    counts_for_json = {
        category: {
            SENTIMENT_NAME[sent]: counts[category][sent]
            for sent in (NEGATIVE, POSITIVE)
        }
        for category in sorted(counts.keys())
    }

    split_counts_for_json = {}
    for (category, sentiment), split_dict in split_counts.items():
        key = f"{category}::{SENTIMENT_NAME[sentiment]}"
        split_counts_for_json[key] = split_dict

    metadata = {
        "dataset_name": DATASET_NAME,
        "selected_categories": [
            {"category": category, "source_file": category_file}
            for category, category_file in selected
        ],
        "topic_label_map": category_topic_to_label,
        "sample_per_category": sample_per_category,
        "raw_empirical_counts": counts_for_json,
        "rarest_cell": {
            "category": rarest_category,
            "sentiment_label": int(rarest_sentiment),
            "sentiment_name": SENTIMENT_NAME[rarest_sentiment],
            "count": int(rarest_count),
            "reason": "minimum non-zero empirical count among category x sentiment cells",
        },
        "split_counts": split_counts_for_json,
        "num_train_docs": len(train_rows),
        "num_eval_docs": len(eval_rows),
        "num_withheld_docs": len(withheld_rows),
        "eval_fraction": eval_fraction,
        "seed": seed,
    }

    with paths.metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class AmazonReviewJsonlDataset(Dataset):
    """Map-style dataset backed by JSONL rows."""

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.rows = _read_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "text": row["text"],
            "topic_label": int(row["topic_label"]),
            "sentiment_label": int(row["sentiment_label"]),
            "topic": row["topic"],
        }


class BalancedLabelBatchSampler(Sampler[list[int]]):
    """Class-balanced batch sampler with optional DDP rank sharding.

    Per batch, each class contributes roughly `batch_size / num_classes` examples.
    Minority classes are oversampled by cycling through shuffled class pools.
    """

    def __init__(
        self,
        dataset: AmazonReviewJsonlDataset,
        label_type: str,
        batch_size: int,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        if label_type not in {"topic", "sentiment"}:
            raise ValueError("label_type must be 'topic' or 'sentiment'")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError("rank must satisfy 0 <= rank < world_size")

        self.dataset = dataset
        self.label_type = label_type
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = drop_last
        self.epoch = 0

        label_key = "topic_label" if label_type == "topic" else "sentiment_label"
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(self.dataset.rows):
            class_to_indices[int(row[label_key])].append(i)

        self.classes = sorted(class_to_indices.keys())
        if len(self.classes) < 2:
            raise RuntimeError(f"Need at least 2 classes for balanced sampling; got {self.classes}")
        if self.batch_size < len(self.classes):
            raise ValueError(
                f"batch_size={self.batch_size} must be >= num_classes={len(self.classes)} "
                "for balanced per-class allocation"
            )
        self.class_to_indices_global = dict(class_to_indices)

        self._rank_class_indices: dict[int, list[int]] = {}
        self._num_batches = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        rank_class_indices: dict[int, list[int]] = {}
        total_rank_items = 0

        for cls in self.classes:
            pool = self.class_to_indices_global[cls].copy()
            rng.shuffle(pool)
            shard = pool[self.rank :: self.world_size]
            # Fallback for tiny classes where this rank got no shard.
            if not shard:
                shard = pool.copy()
            rank_class_indices[cls] = shard
            total_rank_items += len(shard)

        if total_rank_items == 0:
            raise RuntimeError("No rank-local samples available for balanced sampling")

        if self.drop_last:
            self._num_batches = max(1, total_rank_items // self.batch_size)
        else:
            self._num_batches = max(1, math.ceil(total_rank_items / self.batch_size))

        self._rank_class_indices = rank_class_indices

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + 100_003 * self.epoch + self.rank)

        class_pools = {cls: pool.copy() for cls, pool in self._rank_class_indices.items()}
        cursors = {cls: 0 for cls in self.classes}

        for _ in range(self._num_batches):
            class_order = self.classes.copy()
            rng.shuffle(class_order)

            base = self.batch_size // len(class_order)
            remainder = self.batch_size % len(class_order)

            batch_indices: list[int] = []
            for i, cls in enumerate(class_order):
                need = base + (1 if i < remainder else 0)
                pool = class_pools[cls]
                if not pool:
                    continue
                for _ in range(need):
                    if cursors[cls] >= len(pool):
                        rng.shuffle(pool)
                        cursors[cls] = 0
                    batch_indices.append(pool[cursors[cls]])
                    cursors[cls] += 1

            if len(batch_indices) < self.batch_size:
                if self.drop_last:
                    continue
                while len(batch_indices) < self.batch_size:
                    cls = rng.choice(class_order)
                    pool = class_pools[cls]
                    if not pool:
                        break
                    if cursors[cls] >= len(pool):
                        rng.shuffle(pool)
                        cursors[cls] = 0
                    batch_indices.append(pool[cursors[cls]])
                    cursors[cls] += 1

            rng.shuffle(batch_indices)
            yield batch_indices[: self.batch_size]


class BalancedJointLabelBatchSampler(Sampler[list[int]]):
    """Cross-product balanced sampler over (topic_label, sentiment_label) cells.

    This keeps in-batch diversity for *both* labels simultaneously by allocating each
    batch approximately evenly across observed topic x sentiment cells. Minority cells
    are oversampled by cycling shuffled per-cell pools.
    """

    def __init__(
        self,
        dataset: AmazonReviewJsonlDataset,
        batch_size: int,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError("rank must satisfy 0 <= rank < world_size")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = drop_last
        self.epoch = 0

        cell_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, row in enumerate(self.dataset.rows):
            topic = int(row["topic_label"])
            sentiment = int(row["sentiment_label"])
            cell_to_indices[(topic, sentiment)].append(i)

        self.cells = sorted(cell_to_indices.keys())
        if len(self.cells) < 2:
            raise RuntimeError(f"Need at least 2 topic/sentiment cells for joint balancing; got {self.cells}")
        self.cell_to_indices_global = dict(cell_to_indices)

        self._rank_cell_indices: dict[tuple[int, int], list[int]] = {}
        self._num_batches = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        rank_cell_indices: dict[tuple[int, int], list[int]] = {}
        total_rank_items = 0

        for cell in self.cells:
            pool = self.cell_to_indices_global[cell].copy()
            rng.shuffle(pool)
            shard = pool[self.rank :: self.world_size]
            if not shard:
                shard = pool.copy()
            rank_cell_indices[cell] = shard
            total_rank_items += len(shard)

        if total_rank_items == 0:
            raise RuntimeError("No rank-local samples available for joint balanced sampling")

        if self.drop_last:
            self._num_batches = max(1, total_rank_items // self.batch_size)
        else:
            self._num_batches = max(1, math.ceil(total_rank_items / self.batch_size))
        self._rank_cell_indices = rank_cell_indices

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + 100_019 * self.epoch + self.rank)

        cell_pools = {cell: pool.copy() for cell, pool in self._rank_cell_indices.items()}
        cursors = {cell: 0 for cell in self.cells}

        for _ in range(self._num_batches):
            cell_order = self.cells.copy()
            rng.shuffle(cell_order)

            base = self.batch_size // len(cell_order)
            remainder = self.batch_size % len(cell_order)

            batch_indices: list[int] = []
            for i, cell in enumerate(cell_order):
                need = base + (1 if i < remainder else 0)
                pool = cell_pools[cell]
                if not pool:
                    continue
                for _ in range(need):
                    if cursors[cell] >= len(pool):
                        rng.shuffle(pool)
                        cursors[cell] = 0
                    batch_indices.append(pool[cursors[cell]])
                    cursors[cell] += 1

            if len(batch_indices) < self.batch_size:
                if self.drop_last:
                    continue
                while len(batch_indices) < self.batch_size:
                    cell = rng.choice(cell_order)
                    pool = cell_pools[cell]
                    if not pool:
                        break
                    if cursors[cell] >= len(pool):
                        rng.shuffle(pool)
                        cursors[cell] = 0
                    batch_indices.append(pool[cursors[cell]])
                    cursors[cell] += 1

            rng.shuffle(batch_indices)
            yield batch_indices[: self.batch_size]


def collate_document_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "text": [row["text"] for row in batch],
        "topic_label": torch.tensor([row["topic_label"] for row in batch], dtype=torch.long),
        "sentiment_label": torch.tensor([row["sentiment_label"] for row in batch], dtype=torch.long),
        "topic": [row["topic"] for row in batch],
    }


def infinite_dataloader_batches(dataloader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in dataloader:
            yield batch


def load_prepared_datasets(
    output_dir: str | Path,
) -> tuple[AmazonReviewJsonlDataset, AmazonReviewJsonlDataset, AmazonReviewJsonlDataset, dict[str, Any]]:
    paths = prepared_paths(output_dir)
    with paths.metadata_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_ds = AmazonReviewJsonlDataset(paths.train_jsonl)
    eval_ds = AmazonReviewJsonlDataset(paths.eval_jsonl)
    withheld_ds = AmazonReviewJsonlDataset(paths.withheld_jsonl)
    return train_ds, eval_ds, withheld_ds, metadata


def format_count_table(metadata: dict[str, Any]) -> str:
    lines = []
    lines.append("Empirical category x sentiment counts (before withholding):")
    for category in sorted(metadata["raw_empirical_counts"].keys()):
        neg = metadata["raw_empirical_counts"][category]["negative"]
        pos = metadata["raw_empirical_counts"][category]["positive"]
        total = neg + pos
        lines.append(f"  - {category}: negative={neg}, positive={pos}, total={total}")

    rc = metadata["rarest_cell"]
    lines.append(f"Rarest cell: ({rc['category']}, {rc['sentiment_name']}) count={rc['count']}")
    lines.append(
        "Split sizes: "
        f"train={metadata['num_train_docs']}, eval={metadata['num_eval_docs']}, "
        f"withheld={metadata['num_withheld_docs']}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 splits for KronSAE experiments")
    parser.add_argument("--output_dir", type=str, default="data/amazon_reviews")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--sample_per_category", type=int, default=50_000)
    parser.add_argument("--eval_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    metadata = prepare_amazon_reviews(
        output_dir=args.output_dir,
        categories=args.categories,
        num_categories=args.num_categories,
        sample_per_category=args.sample_per_category,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )

    print(format_count_table(metadata))
    print(f"Saved metadata to: {prepared_paths(args.output_dir).metadata_json}")


if __name__ == "__main__":
    main()
