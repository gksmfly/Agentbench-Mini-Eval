"""
Additional analysis beyond GP/SGA: tokens, estimated cost, wall-clock speed,
rounds used, and response verbosity, from raw run logs, for all 3 models
compared in this project.

Mirrors project_db/src/analyze.py's approach and pricing table. Unlike
evaluate.py (which reads output.result.history, a flat list of "Round N: ..."
strings used purely for scoring), this script reads output.history -- the same
role-tagged {role, content} shape used in project_db -- which is the solver's
actual multi-round conversation with the (fixed gpt-3.5-turbo) host. That's the
side of the interaction whose token/cost footprint this script measures.

Usage:
    python src/analyze.py

Inputs:
    results/raw/LTP_runs_<model>.jsonl   (same raw logs evaluate.py uses)

Output:
    results/analysis.csv
"""
import csv
import json
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
OUT_FILE = ROOT / "results" / "analysis.csv"

MODELS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.1-8b", "vicuna-13b-local", "claude-sonnet-5"]

# Same pricing table as project_db/src/analyze.py -- see that file for sourcing notes.
# llama-3.1-8b and vicuna-13b-local were both run locally on GPU (no API), so their
# "cost" is $0 regardless of the token counts below -- the pricing entries exist only
# so the table doesn't KeyError.
# claude-sonnet-5 uses Anthropic's introductory pricing (in effect through 2026-08-31);
# standard pricing is $3.00/$15.00.
PRICING_PER_1M = {
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "llama-3.1-8b": {"input": 0.0, "output": 0.0},
    "vicuna-13b-local": {"input": 0.0, "output": 0.0},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}

# tiktoken has no llama tokenizer; cl100k_base is used as a reasonable cross-model
# approximation for relative comparison (absolute counts for llama will be approximate).
ENCODING = tiktoken.get_encoding("cl100k_base")


def load_runs(path: Path):
    runs = []
    with open(path) as f:
        for line in f:
            runs.append(json.loads(line))
    return runs


def token_and_cost_stats(runs, model):
    total_input_tokens = 0
    total_output_tokens = 0
    for r in runs:
        history = r["output"].get("history", [])
        cum = 0
        for msg in history:
            n = len(ENCODING.encode(msg["content"] or ""))
            if msg["role"] == "agent":
                total_output_tokens += n
                total_input_tokens += cum  # prompt sent to get this response = everything before it
            cum += n
    price = PRICING_PER_1M[model]
    cost = (total_input_tokens / 1e6) * price["input"] + (total_output_tokens / 1e6) * price["output"]
    return total_input_tokens, total_output_tokens, cost


def wall_clock_minutes(runs):
    timestamps = [r["time"]["timestamp"] for r in runs if "time" in r]
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)) / 1000 / 60


def avg_agent_response_chars(runs):
    lengths = []
    for r in runs:
        for m in r["output"].get("history", []):
            if m["role"] == "agent":
                lengths.append(len(m["content"] or ""))
    return sum(lengths) / len(lengths) if lengths else 0


def avg_rounds_used(runs):
    rounds = [r["output"]["result"]["finish_round"] for r in runs if r["output"].get("result")]
    return sum(rounds) / len(rounds) if rounds else 0


def host_irrelevant_rate(runs):
    """Fraction of the fixed host's yes/no/irrelevant answers that were "Irrelevant" --
    a proxy for how often the solver's question missed the point (the framework's
    ENPrompter treats anything that isn't a clear yes/no match as irrelevant)."""
    irrelevant, total = 0, 0
    for r in runs:
        for m in r["output"].get("history", []):
            if m["role"] == "user" and m["content"] and m["content"].strip():
                total += 1
                if m["content"].strip().lower().startswith("irrelevant"):
                    irrelevant += 1
    return irrelevant / total if total else 0


def analyze_model(model: str) -> list:
    runs = load_runs(RAW_DIR / f"LTP_runs_{model}.jsonl")
    rows = []

    in_tok, out_tok, cost = token_and_cost_stats(runs, model)
    rows.append({"model": model, "metric": "total_input_tokens_est", "value": in_tok})
    rows.append({"model": model, "metric": "total_output_tokens_est", "value": out_tok})
    rows.append({"model": model, "metric": "estimated_cost_usd_dev_subset", "value": round(cost, 4)})

    minutes = wall_clock_minutes(runs)
    rows.append({"model": model, "metric": "wall_clock_minutes_dev_subset", "value": minutes})
    if minutes:
        rows.append({"model": model, "metric": "games_per_minute", "value": len(runs) / minutes})

    rows.append({"model": model, "metric": "avg_rounds_used", "value": avg_rounds_used(runs)})
    rows.append({"model": model, "metric": "host_irrelevant_answer_rate", "value": host_irrelevant_rate(runs)})
    rows.append({"model": model, "metric": "avg_agent_response_chars", "value": avg_agent_response_chars(runs)})

    return rows


def main():
    all_rows = []
    for model in MODELS:
        all_rows.extend(analyze_model(model))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUT_FILE}\n")
    for model in MODELS:
        print(f"=== {model} ===")
        for r in all_rows:
            if r["model"] != model:
                continue
            v = r["value"]
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"{r['metric']:<35}{v_str}")
        print()


if __name__ == "__main__":
    main()
