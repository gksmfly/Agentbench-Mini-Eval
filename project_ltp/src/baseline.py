"""
Minimal, single-shot reproduction of AgentBench's LTP (Lateral Thinking Puzzle) task -
no Docker, no THUDM/AgentBench framework, no 25-round solver loop, no GPT-3.5 judge
pipeline for key-point matching.

This exists to show the *core idea* of the paper's LTP environment in the smallest
possible form: give an LLM the host role (it knows the story AND the truth), hand it
one candidate yes/no question from a solver, and see whether it answers "Yes",
"No", or "Irrelevant" in the same multi-turn prompt scaffold AgentBench uses.
It does NOT run the full question-asking loop or grade Game Progress / Single Game
Accuracy - that full loop (25 rounds, real multi-round solver, GPT-3.5-turbo judging
key-point hits) is what the actual AgentBench framework run in `results/raw/` already
did across the 20 dev-split puzzles.

Usage:
    export OPENAI_API_KEY=sk-...
    export GROQ_API_KEY=gsk_...              # only needed for the llama-3.1-8b model
    python src/baseline.py                   # first sample_cases.jsonl entry, gpt-3.5-turbo
    python src/baseline.py --index 3
    python src/baseline.py --model gpt-4o
    python src/baseline.py --model llama-3.1-8b-instant
"""
import argparse
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "sample_cases.jsonl"

# Verbatim from THUDM/AgentBench src/server/tasks/ltp/task.py (`ENPrompter.rules`) -
# this is the rulebook the framework gives the HOST role (the LLM that knows the truth
# and must answer the solver's yes/no/irrelevant questions).
HOST_RULES = """1. You know both the "story" and the "truth". When a user wants to play Lateral Thinking Puzzle, you provide them with the "story". The user only knows the "story" and is unaware of the "truth".
2. The user asks questions that can be answered with "yes," "no," or "irrelevant". Their questions are aimed at guessing the "truth". Based on the "truth", you respond to the user's questions using "yes," "no," or "irrelevant" to guide them towards guessing the correct truth.
3. If the user directly asks for details about the truth using the form of "why" questions, inform them that they need to make their own guesses.
4. You must fully understand and accurately interpret the information from the truth. Based on the information of the truth and the user's past questions, you answer the user's questions. The user's questions may not necessarily contain information from the truth, but your responses must align with the facts of the truth.
5. You can only answer "irrelevant" when the truth cannot provide a direct or indirect answer. Note that this is the only condition for responding "irrelevant"; otherwise, you should answer "yes" or "no."
6. You cannot directly disclose the information from the truth to the user, even if they ask directly.
7. You need to judge the user's questions as a whole and understand their overall intent. Avoid answering based solely on a particular point; your responses must align with the facts of the truth.
8. During the user's process of guessing the truth, if they come close to some truths but still have gaps in understanding the complete truth of the truth, you can provide certain entry point hints. However, you cannot directly reveal information from the truth."""


def load_sample(index: int) -> dict:
    with open(DATA_FILE) as f:
        lines = f.readlines()
    return json.loads(lines[index])


def build_host_history(entry: dict, question: str) -> list[dict]:
    """Reproduces ENPrompter.hosting_instruction()'s 8-turn scaffold (story/truth
    hand-off + key-point recap + rules recap) followed by one solver question,
    exactly as `hosting_append_messages(known=[], solver=question)` would for round 1."""
    story, truth = entry["story"], entry["truth"]
    story_key, answer_key = entry["story_key"], entry["answer_key"]
    return [
        {
            "role": "user",
            "content": f"I need you to be the host of a game called Lateral Thinking Puzzle.\n\n"
            f"Lateral Thinking Puzzle is a game consist of a story and a truth. Your story is: '{story}'\n"
            f"Your truth is: '{truth}'\n\nHere are the game rules:\n{HOST_RULES}\n\n"
            f"During the game process, please adhere to the above game rules to ensure a positive gaming "
            f"experience for the users. Pay close attention to the questions asked and ensure that your "
            f"responses comply with both the game rules and the information from the truth. When a user "
            f'requests to play the game, provide them with the story and help them guess the truth by '
            f'answering with "yes", "no", or "irrelevant". Remember that with each response, you must '
            f"fully understand and abide by the aforementioned game rules, as well as the story and the "
            f"truth. This will ensure a smooth user experience and avoid situations where you cannot "
            f"answer or violate the game rules.",
        },
        {
            "role": "assistant",
            "content": "Alright, I understand that my role is to be the host of the Lateral Thinking Puzzle and "
            "help users guess the truth by answering their questions. I have fully grasped all the "
            "information regarding the story and the truth and have carefully read all the rules. I "
            "assure that I will abide by all the rules throughout the game process.",
        },
        {"role": "user", "content": "Please summarize the key points of the story to ensure that you have understood it."},
        {"role": "assistant", "content": story_key},
        {"role": "user", "content": "Please summarize the key points of the truth to ensure that you have understood it."},
        {"role": "assistant", "content": answer_key},
        {"role": "user", "content": "Please restate the rules to ensure that you have understood all of them."},
        {"role": "assistant", "content": HOST_RULES},
        {
            "role": "user",
            "content": "Alright, we can now start the game. Remember, before each response, you should review the "
            'key points of the story, the key points of the truth, and the rules. Answer with "yes", '
            '"no", or "irrelevant".',
        },
        {
            "role": "assistant",
            "content": f"Alright, as the host of the game, I will adhere to the above rules and ensure that my "
            f"responses comply with the rules and the information from the truth. Below is your story: "
            f"\n{story}\n\nYou can start guessing the content of the truth, and I will answer your "
            f'questions. Please note that your questions should be answerable with "yes", "no", '
            f'or "irrelevant".',
        },
        {"role": "user", "content": f'{question}\nPlease answer with "yes", "no", or "irrelevant".'},
    ]


def call_llm(model: str, history: list[dict]) -> str:
    """Routes to OpenAI for gpt-* models, Groq for llama-* models (both OpenAI-compatible APIs)."""
    if model.startswith("llama"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("Set GROQ_API_KEY first: export GROQ_API_KEY=gsk_...")
    else:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY first: export OPENAI_API_KEY=sk-...")

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "temperature": 0, "messages": history},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    entry = load_sample(args.index)
    question = entry["sample_question"]
    gold = entry.get("gold_host_answer")

    print(f"[puzzle index] {entry['index']}")
    print(f"[story]  {entry['story']}\n")
    print(f"[question put to the host] {question}\n")

    history = build_host_history(entry, question)
    reply = call_llm(args.model, history)

    print(f"[{args.model} host response]\n{reply}\n")
    print(f"[gold host answer for reference, captured live by the framework] {gold}")
    print(
        "\nNote: this script only asks the host ONE question and reads back its verdict.\n"
        "It does not run the 25-round solver loop, does not call a judge model to check\n"
        "whether the question repeats the story or a prior question, and does not score\n"
        "Game Progress / SGA - see src/evaluate.py and results/raw/ for the full run."
    )


if __name__ == "__main__":
    main()
