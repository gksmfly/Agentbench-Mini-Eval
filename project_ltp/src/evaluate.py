"""
Recompute AgentBench's LTP (Lateral Thinking Puzzle) metrics - Game Progress (GP),
Single Game Accuracy (SGA), Round Efficiency (RE), Query Relevance (QR) - from raw
run logs, for all 3 models compared in this project.

This reimplements the exact scoring rule used by AgentBench's
`src/server/tasks/ltp/task.py` (`LateralThinkingPuzzle.start_sample`, THUDM/AgentBench),
so the numbers here match each raw run's own `output.result` fields (produced live by
the framework during the actual 25-round host/solver loop) almost exactly - this
script exists to make that scoring logic transparent and independently checkable
from the flat interaction transcript alone, not to replace it.

Independent recomputation, no LLM calls:
  - SGA (accuracy) and QR (relevance) are recovered by re-parsing each round's
    "Round N: <question>" / "<host answer>" pair out of the logged transcript
    (`result.history`) and re-applying the framework's own `check_yes` / `check_no`
    string-prefix rules - no need to re-ask any model anything.
  - RE (efficiency) is recomputed from the logged `finish_round` and the fixed
    `rounds` limit (25, see configs/tasks/ltp.yaml).
  - GP (progress) is recomputed as len(hit_keys) / hints, where hit_keys is the
    framework's own logged dict of matched key points and hints is the original
    number of gold key-point lines for that puzzle (data/ltp_dev_gold.jsonl).

Usage:
    python src/evaluate.py

Inputs (checked in by default, see ../data and ../results/raw):
    data/ltp_dev_gold.jsonl                       gold key-point counts (20 dev puzzles)
    results/raw/LTP_runs_<model>.jsonl            raw per-sample outputs from each real run

Output:
    results/metrics.csv   (one row per metric per model, plus a recomputed-vs-logged check)
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT / "data" / "ltp_dev_gold.jsonl"
RAW_DIR = ROOT / "results" / "raw"
OUT_FILE = ROOT / "results" / "metrics.csv"

MODELS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.1-8b"]
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
            runs[d["index"]] = d["output"]["result"]
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


def evaluate_model(model: str, gold: dict) -> list:
    runs = load_runs(RAW_DIR / f"LTP_runs_{model}.jsonl")

    rows = []
    logged_totals = {"SGA": [], "RE": [], "QR": [], "GP": []}
    recomputed_totals = {"SGA": [], "RE": [], "QR": [], "GP": []}
    mismatches = 0

    for index, result in sorted(runs.items()):
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

        row_mismatch = False
        for metric in ("SGA", "RE", "QR", "GP"):
            logged_totals[metric].append(logged[metric])
            recomputed_totals[metric].append(recomputed[metric])
            if abs(logged[metric] - recomputed[metric]) > 1e-6:
                row_mismatch = True
        if row_mismatch:
            mismatches += 1

        rows.append(
            {
                "model": model,
                "index": index,
                "SGA_logged": logged["SGA"], "SGA_recomputed": recomputed["SGA"],
                "RE_logged": logged["RE"], "RE_recomputed": recomputed["RE"],
                "QR_logged": logged["QR"], "QR_recomputed": recomputed["QR"],
                "GP_logged": logged["GP"], "GP_recomputed": recomputed["GP"],
                "match": not row_mismatch,
            }
        )

    n = len(runs)
    overall = {
        "model": model,
        "index": "OVERALL (mean)",
        "SGA_logged": sum(logged_totals["SGA"]) / n, "SGA_recomputed": sum(recomputed_totals["SGA"]) / n,
        "RE_logged": sum(logged_totals["RE"]) / n, "RE_recomputed": sum(recomputed_totals["RE"]) / n,
        "QR_logged": sum(logged_totals["QR"]) / n, "QR_recomputed": sum(recomputed_totals["QR"]) / n,
        "GP_logged": sum(logged_totals["GP"]) / n, "GP_recomputed": sum(recomputed_totals["GP"]) / n,
        "match": mismatches == 0,
    }
    rows.append(overall)
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
    fieldnames = [
        "model", "index",
        "SGA_logged", "SGA_recomputed",
        "RE_logged", "RE_recomputed",
        "QR_logged", "QR_recomputed",
        "GP_logged", "GP_recomputed",
        "match",
    ]
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {OUT_FILE}\n")
    for model in MODELS:
        print(f"=== {model} ===")
        overall = next(r for r in all_rows if r["model"] == model and r["index"] == "OVERALL (mean)")
        for metric in ("SGA", "RE", "QR", "GP"):
            print(
                f"{metric:<5} logged={overall[f'{metric}_logged']:.4f}  "
                f"recomputed={overall[f'{metric}_recomputed']:.4f}"
            )
        print()

    print("--- verification summary (recomputed from raw transcripts vs. framework's own logged scores) ---")
    for model, n, mismatches in summary:
        status = "MATCH" if mismatches == 0 else f"{mismatches}/{n} SAMPLES DIFFER"
        print(f"{model:<16} n={n:<3} {status}")


if __name__ == "__main__":
    main()
