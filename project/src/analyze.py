"""
Additional analysis beyond accuracy: tokens, estimated cost, wall-clock speed,
SQL execution error rate, response verbosity, and where failures concentrate.

This is deliberately separate from evaluate.py (which reproduces AgentBench's own
accuracy metric) — this script answers "how expensive/fast/robust was each model",
not "was it correct".

Usage:
    python src/analyze.py

Inputs:
    results/raw/dbbench_std_runs_<model>.jsonl   (same raw logs evaluate.py uses)

Output:
    results/analysis.csv
"""
import csv
import json
import re
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
OUT_FILE = ROOT / "results" / "analysis.csv"

MODELS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.1-8b"]

# Approximate public per-1M-token pricing at time of writing. Not guaranteed current —
# check the provider's pricing page before using these numbers for a real budget.
# llama-3.1-8b was actually run via Groq's free tier, so real spend was $0; the paid
# rate is included only so the "if you weren't on the free tier" comparison is visible.
PRICING_PER_1M = {
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "llama-3.1-8b": {"input": 0.05, "output": 0.08},
}

# tiktoken has no llama tokenizer; cl100k_base is used as a reasonable cross-model
# approximation for relative comparison (absolute counts for llama will be approximate).
ENCODING = tiktoken.get_encoding("cl100k_base")

SQL_ERROR_PATTERN = re.compile(r"\b(ERROR|1064|1146|1054|1305|1planning)\b", re.IGNORECASE)


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


def sql_error_rate(runs):
    """Fraction of samples where MySQL returned an error to the agent at least once
    during the interaction (regardless of whether the sample ultimately succeeded)."""
    with_error = 0
    for r in runs:
        history = r["output"].get("history", [])
        if any(m["role"] == "user" and SQL_ERROR_PATTERN.search(m["content"] or "") for m in history):
            with_error += 1
    return with_error / len(runs) if runs else 0


def avg_agent_response_chars(runs):
    lengths = []
    for r in runs:
        for m in r["output"].get("history", []):
            if m["role"] == "agent":
                lengths.append(len(m["content"] or ""))
    return sum(lengths) / len(lengths) if lengths else 0


def status_by_type(runs):
    """Cross-tab: for each SQL type bucket, what fraction completed vs failed."""
    buckets = {}
    for r in runs:
        result = r["output"].get("result") or {}
        t = result.get("type", "unknown")
        status = r["output"]["status"]
        buckets.setdefault(t, {}).setdefault(status, 0)
        buckets[t][status] += 1
    return buckets


def analyze_model(model: str) -> list:
    runs = load_runs(RAW_DIR / f"dbbench_std_runs_{model}.jsonl")
    rows = []

    in_tok, out_tok, cost = token_and_cost_stats(runs, model)
    rows.append({"model": model, "metric": "total_input_tokens_est", "value": in_tok})
    rows.append({"model": model, "metric": "total_output_tokens_est", "value": out_tok})
    rows.append({"model": model, "metric": "estimated_cost_usd_300_samples", "value": round(cost, 4)})
    rows.append({"model": model, "metric": "estimated_cost_usd_per_correct_answer",
                 "value": None})  # filled in by main() once accuracy is known

    minutes = wall_clock_minutes(runs)
    rows.append({"model": model, "metric": "wall_clock_minutes_300_samples", "value": minutes})
    if minutes:
        rows.append({"model": model, "metric": "samples_per_minute", "value": len(runs) / minutes})

    rows.append({"model": model, "metric": "sql_execution_error_rate", "value": sql_error_rate(runs)})
    rows.append({"model": model, "metric": "avg_agent_response_chars", "value": avg_agent_response_chars(runs)})

    return rows


# Success Rate per model, copied from evaluate.py's output so cost-per-correct-answer
# can be computed without re-deriving accuracy here.
SUCCESS_RATE = {
    "gpt-3.5-turbo": 0.4467,
    "gpt-4o": 0.4733,
    "llama-3.1-8b": 0.0,
}


def main():
    all_rows = []
    for model in MODELS:
        rows = analyze_model(model)
        n_correct = SUCCESS_RATE[model] * 300
        for row in rows:
            if row["metric"] == "estimated_cost_usd_per_correct_answer":
                cost_row = next(r for r in rows if r["metric"] == "estimated_cost_usd_300_samples")
                row["value"] = round(cost_row["value"] / n_correct, 4) if n_correct > 0 else None
        all_rows.extend(rows)

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
            print(f"{r['metric']:<40}{v_str}")
        print()

    print("--- status x SQL type breakdown ---")
    for model in MODELS:
        runs = load_runs(RAW_DIR / f"dbbench_std_runs_{model}.jsonl")
        print(f"\n{model}:")
        for typ, statuses in status_by_type(runs).items():
            print(f"  {typ}: {statuses}")


if __name__ == "__main__":
    main()
