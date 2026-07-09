"""
Recompute AgentBench's LTP (Lateral Thinking Puzzle) metrics - Game Progress (GP),
Single Game Accuracy (SGA), Round Efficiency (RE), Query Relevance (QR) - from raw
run logs, for all 3 models compared in this project.

Mirrors project_db/src/evaluate.py's shape: reads raw logs directly (no intermediate
per-game summary CSV), independently reimplements the framework's own scoring rule,
and writes only the aggregate metrics to results/metrics.csv in the same long format
(model, metric, value, n_samples) that project_db uses.

This reimplements the exact scoring rule used by AgentBench's
`src/server/tasks/ltp/task.py` (`LateralThinkingPuzzle.start_sample`, THUDM/AgentBench):
  - SGA (accuracy) and QR (relevance) are recovered by re-parsing each round's
    "Round N: <question>" / "<host answer>" pair out of the logged transcript
    (`result.history`) and re-applying the framework's own `check_yes` / `check_no`
    string-prefix rules - no need to re-ask any model anything.
  - RE (efficiency) is recomputed from the logged `finish_round` and the fixed
    `rounds` limit (25, see configs/tasks/ltp.yaml).
  - GP (progress) is recomputed as len(hit_keys) / hints, where hit_keys is the
    framework's own logged dict of matched key points and hints is the original
    number of gold key-point lines for that puzzle (data/ltp_dev_gold.jsonl).

The recomputed-vs-logged comparison (does our independent scoring match what the
framework logged live?) is verification, not a metric to publish -- it's printed to
the console, not written to results/metrics.csv.

Usage:
    python src/evaluate.py

Inputs (checked in by default, see ../data and ../results/raw):
    data/ltp_dev_gold.jsonl                       gold key-point counts (20 dev puzzles)
    results/raw/LTP_runs_<model>.jsonl            raw per-sample outputs from each real run

Output:
    results/metrics.csv   (one row per aggregate metric per model)
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT / "data" / "ltp_dev_gold.jsonl"
RAW_DIR = ROOT / "results" / "raw"
OUT_FILE = ROOT / "results" / "metrics.csv"

MODELS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.1-8b", "vicuna-13b-local", "claude-sonnet-5"]
ROUNDS = 25  # configs/tasks/ltp.yaml: parameters.round

ROUND_LINE = re.compile(r"^Round \d+:")


def load_gold(path: Path) -> dict:
    gold = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            gold[entry["index"]] = entry["hints"]
    return gold


def load_runs(path: Path) -> dict:
    runs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            runs[d["index"]] = {"status": d["output"]["status"], "result": d["output"]["result"]}
    return runs


def check_yes(message: str) -> bool:
    # Verbatim from ENPrompter.check_yes
    message = message.strip()
    return message.startswith("Yes") or message.startswith("yes")


def check_no(message: str) -> bool:
    # Verbatim from ENPrompter.check_no
    message = message.strip()
    return message.startswith("No") or message.startswith("no")


def recompute_from_history(history: list) -> tuple:
    """Re-derive (correct, relevant) round counts by walking the flat transcript:
    every "Round N: ..." line is immediately followed by the host's verbatim answer
    for that round (task.py always logs.append(host) right after logging the question,
    except on the rare early-abort path - which never fires in these transcripts since
    all logged samples ran the full "task limit reached" 25 rounds)."""
    correct, relevant = 0, 0
    for i, line in enumerate(history):
        if not isinstance(line, str) or not ROUND_LINE.match(line):
            continue
        if i + 1 >= len(history):
            continue
        answer = history[i + 1]
        if check_yes(answer):
            correct += 1
            relevant += 1
        elif check_no(answer):
            relevant += 1
    return correct, relevant


def evaluate_model(model: str, gold: dict) -> tuple:
    runs = load_runs(RAW_DIR / f"LTP_runs_{model}.jsonl")
    n = len(runs)

    recomputed_totals = {"SGA": [], "RE": [], "QR": [], "GP": []}
    status_counts = {}
    mismatches = 0

    for index, entry in sorted(runs.items()):
        result = entry["result"]
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

        history = result["history"]
        finish_round = result["finish_round"]
        hit_keys = result["hit_keys"]
        hints = gold[index]

        correct, relevant = recompute_from_history(history)
        recomputed = {
            "SGA": correct / ROUNDS,
            "RE": 1 - finish_round / ROUNDS,
            "QR": relevant / ROUNDS,
            "GP": len(hit_keys) / hints if hints else 0.0,
        }
        logged = {
            "SGA": result["accuracy"],
            "RE": result["efficiency"],
            "QR": result["relevance"],
            "GP": result["progress"],
        }
        if any(abs(logged[m] - recomputed[m]) > 1e-6 for m in recomputed):
            mismatches += 1
        for m in recomputed_totals:
            recomputed_totals[m].append(recomputed[m])

    rows = []
    for metric, values in recomputed_totals.items():
        rows.append({"model": model, "metric": metric, "value": sum(values) / n, "n_samples": n})
    for status, count in status_counts.items():
        rows.append({"model": model, "metric": f"status:{status}", "value": count / n, "n_samples": n})

    return rows, mismatches, n


def main():
    gold = load_gold(GOLD_FILE)

    all_rows = []
    summary = []
    for model in MODELS:
        rows, mismatches, n = evaluate_model(model, gold)
        all_rows.extend(rows)
        summary.append((model, n, mismatches))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "value", "n_samples"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUT_FILE}\n")
    for model in MODELS:
        print(f"=== {model} ===")
        for r in all_rows:
            if r["model"] != model:
                continue
            print(f"{r['metric']:<25}{r['value']:.4f}")
        print()

    print("--- verification (recomputed from raw transcripts vs. framework's own logged scores; not written to metrics.csv) ---")
    for model, n, mismatches in summary:
        status = "MATCH" if mismatches == 0 else f"{mismatches}/{n} SAMPLES DIFFER"
        print(f"{model:<16} n={n:<3} {status}")


if __name__ == "__main__":
    main()
