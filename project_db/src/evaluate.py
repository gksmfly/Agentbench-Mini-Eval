"""
Recompute AgentBench's DB-environment (dbbench) accuracy metrics from raw run logs,
for all 3 models compared in this project.

This reimplements the exact scoring rule used by AgentBench's
`src/server/tasks/dbbench/DBBench.metrics` (THUDM/AgentBench), so the numbers here
match each model's own `results/raw/dbbench_std_overall_<model>.json` (produced live
by the framework) almost exactly — this script exists to make that scoring logic
transparent and independently checkable, not to replace it.

Usage:
    python src/evaluate.py

Inputs (checked in by default, see ../data and ../results/raw):
    data/dbbench_standard_gold.jsonl                  gold labels (300 problems, shared)
    results/raw/dbbench_std_runs_<model>.jsonl         model outputs from each real run

Output:
    results/metrics.csv   (one row per metric per model, "model" column added)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT / "data" / "dbbench_standard_gold.jsonl"
RAW_DIR = ROOT / "results" / "raw"
OUT_FILE = ROOT / "results" / "metrics.csv"

MODELS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.1-8b", "claude-sonnet-5"]

TYPES = [
    "other", "counting", "comparison", "ranking",
    "aggregation-SUM", "aggregation-MIN", "aggregation-MAX", "aggregation-AVG",
    "SELECT", "INSERT", "UPDATE",
]


def load_gold(path: Path):
    gold = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                ans = entry["answer_md5"]
            else:
                ans = entry["label"]
            gold.append(ans)
    return gold


def load_runs(path: Path):
    runs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            runs[d["index"]] = d["output"]
    return runs


def accuracy_for_type(typ, outputs, golds):
    correct, total = 0, 0
    for entry, cor in zip(outputs, golds):
        if not entry:
            continue
        ans, t = entry["answer"], entry["type"]
        if t != typ and not (typ == "SELECT" and t not in ("INSERT", "UPDATE")):
            continue
        if t in ("INSERT", "DELETE", "UPDATE"):
            correct += ans == str(cor)
        else:
            try:
                parsed = list(eval(ans))
            except Exception:
                parsed = [ans]
            if len(parsed) == 1 and len(cor) == 1:
                try:
                    correct += float(parsed[0]) == float(cor[0])
                except (ValueError, TypeError):
                    correct += parsed[0] == cor[0]
            else:
                try:
                    correct += set(parsed) == set(cor)
                except TypeError:
                    pass
        total += 1
    return (correct / total) if total else None, total


def status_breakdown(runs):
    counts = {}
    history_lengths = []
    for out in runs.values():
        status = out["status"]
        counts[status] = counts.get(status, 0) + 1
        history_lengths.append(len(out.get("history", [])))
    total = len(runs)
    breakdown = {k: v / total for k, v in counts.items()}
    avg_history = sum(history_lengths) / len(history_lengths) if history_lengths else 0
    return breakdown, avg_history


def evaluate_model(model: str, gold: list) -> list:
    runs_file = RAW_DIR / f"dbbench_std_runs_{model}.jsonl"
    runs = load_runs(runs_file)
    outputs = [runs[i]["result"] if i in runs else None for i in range(len(gold))]

    rows = []
    for typ in TYPES:
        acc, n = accuracy_for_type(typ, outputs, gold)
        rows.append({"model": model, "metric": f"{typ}_accuracy", "value": acc, "n_samples": n})

    cat_accs = [r["value"] for r in rows if r["metric"] in
                ("SELECT_accuracy", "INSERT_accuracy", "UPDATE_accuracy")]
    overall = sum(cat_accs) / len(cat_accs)
    rows.append({"model": model, "metric": "overall_cat_accuracy (Success Rate)",
                 "value": overall, "n_samples": len(gold)})

    breakdown, avg_history = status_breakdown(runs)
    for status, frac in breakdown.items():
        rows.append({"model": model, "metric": f"status:{status}", "value": frac, "n_samples": len(runs)})
    rows.append({"model": model, "metric": "avg_history_length (proxy for rounds/cost)",
                 "value": avg_history, "n_samples": len(runs)})
    return rows


def main():
    gold = load_gold(GOLD_FILE)

    all_rows = []
    for model in MODELS:
        all_rows.extend(evaluate_model(model, gold))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "value", "n_samples"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUT_FILE}\n")
    for model in MODELS:
        print(f"=== {model} ===")
        print(f"{'metric':<45}{'value'}")
        for r in all_rows:
            if r["model"] != model:
                continue
            v = r["value"]
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"{r['metric']:<45}{v_str}")
        print()


if __name__ == "__main__":
    main()
