"""
Minimal, single-shot reproduction of AgentBench's DB (dbbench) task —
no Docker, no MySQL, no multi-round agent loop, no THUDM/AgentBench framework.

This exists to show the *core idea* of the paper's DB environment in the
smallest possible form: give an LLM a table + a natural-language question,
in the same prompt format AgentBench uses, and see what SQL it proposes.
It does NOT execute the SQL or grade it — that full loop (real MySQL,
multi-round correction, hashing-based grading) is what the actual
AgentBench framework run in `results/` already did across all 300 problems.

Usage:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...      # only needed for the claude-sonnet-5 model
    export HF_TOKEN=hf_...                   # only needed for the llama-3.1-8b model (gated on HF)
    python src/baseline.py                   # first sample_cases.jsonl entry, gpt-3.5-turbo
    python src/baseline.py --index 3
    python src/baseline.py --model gpt-4o
    python src/baseline.py --model claude-sonnet-5
    python src/baseline.py --model llama-3.1-8b            # loads locally on GPU, no API call
"""
import argparse
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "sample_cases.jsonl"

# Verbatim from THUDM/AgentBench src/server/tasks/dbbench/__init__.py (`big_prompt`)
SYSTEM_PROMPT = """
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer.
your operation should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then write down:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Your input will be raw MySQL response, you have to deal with it by yourself.
"""


def load_sample(index: int) -> dict:
    with open(DATA_FILE) as f:
        lines = f.readlines()
    return json.loads(lines[index])


def build_user_prompt(entry: dict) -> str:
    return entry["description"] + "\n" + entry.get("add_description", "")


_LOCAL_MODEL_CACHE = {}


def call_local_llama(history: list[dict]) -> str:
    """Loads meta-llama/Llama-3.1-8B-Instruct on the local GPU (4-bit quantized) and
    generates one reply. This is how llama-3.1-8b was actually run for this reproduction —
    no external API involved. Requires a CUDA GPU + transformers/accelerate/bitsandbytes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    if model_id not in _LOCAL_MODEL_CACHE:
        from huggingface_hub import login

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            device_map="auto",
        )
        _LOCAL_MODEL_CACHE[model_id] = (tokenizer, model)
    tokenizer, model = _LOCAL_MODEL_CACHE[model_id]

    inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def call_llm(model: str, history: list[dict]) -> str:
    """Routes to OpenAI for gpt-* models, Anthropic for claude-* models; llama-* models
    run locally on GPU (see call_local_llama), not through an external API."""
    if model.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=sk-ant-...")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 512, "thinking": {"type": "disabled"}, "messages": history},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    if model.startswith("llama"):
        return call_local_llama(history)

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
    table_name = entry["table"]["table_name"]
    question = build_user_prompt(entry)
    gold = entry.get("label") or entry.get("answer_md5")

    print(f"[table]  {table_name}")
    print(f"[type]   {entry['type']}")
    print(f"[question]\n{question}\n")

    history = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": question},
    ]
    reply = call_llm(args.model, history)

    print(f"[{args.model} response]\n{reply}\n")
    print(f"[gold answer for reference] {gold}")
    print(
        "\nNote: this script only asks the model for ONE SQL proposal.\n"
        "It does not run that SQL or grade it — see src/evaluate.py and\n"
        "results/ for the full 300-sample, real-MySQL, multi-round result."
    )


if __name__ == "__main__":
    main()
