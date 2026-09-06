"""External LLM judge for the neutral-prefix steering generations.

Every sentiment number in sec.19.11-19.12 came from this project's own layer-12 logistic probe --
the same object family as the steering direction, so the metric and the intervention share a
representation and the result is partly self-graded. This script re-judges the same generations
with Qwen3-14B, which has no connection to the dictionary, the probe, or the activation cache.

The judge returns a 1-5 Likert sentiment rating plus a fluency flag, in strict JSON. Ratings are
mapped to [0, 1] as (rating - 1) / 4 so they sit on the same scale as the probe's
P(sentiment-positive) and the two can be plotted against each other.

Run AFTER eval_neutral_generation.py -- it reads that script's output JSON and writes a copy
annotated with `judge_rating`, `judge_norm`, and `judge_fluent` per entry.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "You are a careful annotator. You rate the sentiment of short text continuations. "
    "Answer with strict JSON only, no prose, no markdown fences."
)

TEMPLATE = """A small language model was given the prompt below and produced the continuation below.

Prompt: {prompt!r}
Continuation: {continuation!r}

Rate the SENTIMENT expressed about the thing being described, judging the continuation only.
Use this scale:
1 = strongly negative, 2 = negative, 3 = neutral / no sentiment, 4 = positive, 5 = strongly positive.

Also judge whether the continuation is FLUENT: grammatical, on-topic English that a person could
have written. Degenerate output (repeated tokens, word salad, broken syntax) is not fluent.

Reply with exactly this JSON object and nothing else:
{{"sentiment": <1-5 integer>, "fluent": <true|false>}}"""


def parse_verdict(text: str) -> tuple[float, bool]:
    """Tolerant parse: the model may still wrap the JSON or emit a <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            rating = float(obj["sentiment"])
            return rating, bool(obj.get("fluent", True))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    fallback = re.search(r"[1-5]", text)
    return (float(fallback.group(0)) if fallback else float("nan")), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default=str(ROOT / "runs" / "neutral_generation_results.json"))
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "neutral_generation_judged.json"))
    ap.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    data = json.loads(Path(args.in_path).read_text())

    flat_entries = []
    for cond in data["conditions"]:
        for entry in cond["entries"]:
            flat_entries.append((cond["label"], entry))
    print(f"{len(flat_entries)} generations to judge", flush=True)

    # transformers rather than vLLM: vLLM's Triton path needs a bundled CUDA include dir that this
    # install does not ship (`cuda.h: No such file or directory`), and it ignores CUDA_HOME. Plain
    # transformers generation works on this GPU and 792 short judgements is well within its reach.
    import torch as t
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    tokenizer.padding_side = "left"  # required for correct batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    for _, entry in flat_entries:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(prompt=entry["prompt"], continuation=entry["continuation"])},
        ]
        # enable_thinking=False: a 1-5 rating does not need Qwen3's reasoning trace, and
        # disabling it keeps outputs short and parseable.
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, dtype=t.bfloat16, device_map="cuda:0"
    ).eval()

    verdict_texts: list[str] = []
    bs = args.batch_size
    for start in range(0, len(prompts), bs):
        batch = prompts[start:start + bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_model_len).to("cuda:0")
        with t.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        verdict_texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"  judged {min(start + bs, len(prompts))}/{len(prompts)}", flush=True)

    for (label, entry), text in zip(flat_entries, verdict_texts):
        rating, fluent = parse_verdict(text)
        entry["judge_raw"] = text
        entry["judge_rating"] = rating
        entry["judge_norm"] = (rating - 1.0) / 4.0 if rating == rating else float("nan")
        entry["judge_fluent"] = fluent

    Path(args.out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved to {args.out_path}\n", flush=True)

    print(f"{'condition':<22} {'n':>4} {'judge_norm':>11} {'judge_1-5':>10} {'fluent%':>8} {'probe':>7} {'ppl':>8}")
    for cond in data["conditions"]:
        entries = cond["entries"]
        ratings = [e["judge_rating"] for e in entries if e["judge_rating"] == e["judge_rating"]]
        norms = [e["judge_norm"] for e in entries if e["judge_norm"] == e["judge_norm"]]
        probes = [e["probe_sentiment_p_positive"] for e in entries
                  if e["probe_sentiment_p_positive"] == e["probe_sentiment_p_positive"]]
        ppls = [e["self_perplexity"] for e in entries if e["self_perplexity"] == e["self_perplexity"]]
        fluent_pct = 100.0 * sum(1 for e in entries if e.get("judge_fluent")) / max(len(entries), 1)
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        print(f"{cond['label']:<22} {len(entries):>4} {mean(norms):>11.3f} {mean(ratings):>10.2f} "
              f"{fluent_pct:>7.1f}% {mean(probes):>7.3f} {mean(ppls):>8.2f}")


if __name__ == "__main__":
    main()
