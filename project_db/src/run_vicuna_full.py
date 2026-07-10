"""
vicuna-13b-local을 dbbench-std 300문제 전체로 실행 (원격 GPU 서버용, tmux/nohup 백그라운드 실행 대상).

THUDM/AgentBench 원본 채점 로직을 그대로 이식한 독립 실행 스크립트입니다:
- THUDM/AgentBench의 build_init_sql()/정규식 파싱/MD5 채점 쿼리/상태값을 그대로 이식
- Docker도, THUDM/AgentBench 원본 클론도 필요 없음 -- 이 파일이 그 로직을 전부 독립적으로 재구현
- MySQL도 이 스크립트가 직접 설치/설정 (Colab의 MySQL과는 별개로 이 서버에 새로 설치)
- vicuna는 FastChat vicuna_v1.1 템플릿으로 직접 프롬프트 조립 (apply_chat_template 미지원 모델이라)

사전 준비 (딱 이것만):
    pip install mysql-connector-python transformers accelerate bitsandbytes torch
    sudo -v   # sudo 비밀번호를 미리 캐싱해둬야 스크립트 중간에 멈추지 않습니다

실행 (repo 루트의 project_db/ 안에서, tmux 세션 안에서 실행 권장 -- SSH 끊겨도 안 죽음):
    tmux new -s vicuna
    python src/run_vicuna_full.py --limit 10        # 먼저 소수만 테스트
    python src/run_vicuna_full.py                   # 전체 300문제 (Ctrl+B, D로 detach, 재접속: tmux attach -t vicuna)

중간에 끊겨도 이미 처리된 인덱스는 건너뛰고 이어서 진행합니다 (재실행 시 자동 resume).

출력: results/raw/dbbench_std_runs_vicuna-13b-local.jsonl
      (project_db의 다른 4개 모델과 완전히 같은 포맷 -- 이미 evaluate.py/analyze.py의 MODELS에
      "vicuna-13b-local"이 추가되어 5번째 모델로 합류한 상태)
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "dbbench_standard_full.jsonl"
OUT_FILE = ROOT / "results" / "raw" / "dbbench_std_runs_vicuna-13b-local.jsonl"

MODEL_ID = "lmsys/vicuna-13b-v1.5"
MAX_ROUND = 15
MYSQL_PASSWORD = "password"

# THUDM/AgentBench src/server/tasks/dbbench/__init__.py의 big_prompt 원문 그대로
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

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def setup_mysql():
    """MySQL을 설치(이미 있으면 건너뜀)하고 root 비밀번호를 설정한 뒤 접속 가능한 상태로 만듭니다."""
    import mysql.connector

    try:
        conn = mysql.connector.connect(host="127.0.0.1", user="root", password=MYSQL_PASSWORD, port=3306)
        conn.close()
        print("MySQL 이미 설정되어 있음, 건너뜀")
        return
    except Exception:
        pass

    print("MySQL 설치/설정 중 (sudo 필요)...")
    subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "mysql-server"], check=True)
    subprocess.run(["sudo", "service", "mysql", "start"], check=True)
    subprocess.run(
        ["sudo", "mysql", "-e",
         f"ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '{MYSQL_PASSWORD}'; FLUSH PRIVILEGES;"],
        check=True,
    )
    print("MySQL 준비 완료")


def build_init_sql(entry):
    """AgentBench/src/server/tasks/dbbench/__init__.py의 build_init_sql() 그대로 이식."""
    name = entry["table"]["table_name"]
    columns = ",".join([f"`{c['name']}` TEXT" for c in entry["table"]["table_info"]["columns"]])
    column_names = ",".join([f"`{c['name']}`" for c in entry["table"]["table_info"]["columns"]])
    items = []
    items_data = ()
    for row in entry["table"]["table_info"]["rows"]:
        item = "("
        for col in row:
            item += "%s,"
            items_data += (col,)
        item = item[:-1] + ")"
        items.append(item)
    items = ",".join(items)
    sql = f"""CREATE DATABASE IF NOT EXISTS `{name}`;
USE `{name}`;
CREATE TABLE IF NOT EXISTS `{name}` ({columns});
INSERT INTO `{name}` ({column_names}) VALUES {items};
COMMIT;
"""
    return sql, items_data


class Container:
    """AgentBench/src/server/tasks/dbbench/Interaction.py의 Container 그대로 이식,
    Docker 컨테이너 대신 이 서버에 직접 설치한 MySQL(127.0.0.1:3306)에 연결."""

    def __init__(self):
        import mysql.connector

        self.conn = mysql.connector.connect(
            host="127.0.0.1", user="root", password=MYSQL_PASSWORD, port=3306,
            pool_reset_session=True,
        )

    def execute(self, sql, database=None, data=()):
        self.conn.reconnect()
        try:
            with self.conn.cursor() as cursor:
                if database:
                    cursor.execute(f"use `{database}`;")
                    cursor.fetchall()
                cursor.execute(sql, data, multi=True)
                result = cursor.fetchall()
                result = str(result)
            self.conn.commit()
        except Exception as e:
            result = str(e)
        if len(result) > 800:
            result = result[:800] + "[TRUNCATED]"
        return result


def build_vicuna_prompt(history):
    prompt = VICUNA_SYSTEM + " "
    for m in history:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        prompt += f"{role}: {m['content']}" + (" " if role == "USER" else "</s>")
    return prompt + "ASSISTANT:"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("GPU 사용 가능:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(없음, CPU로 진행 시 매우 느릴 수 있음)")
    print("모델 로드 중...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    print("모델 로드 완료")
    return tokenizer, model


def generate_vicuna(tokenizer, model, history, max_new_tokens=512):
    prompt = build_vicuna_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def run_sample(entry, container, tokenizer, model):
    """AgentBench/src/server/tasks/dbbench/__init__.py의 DBBench.start_sample() 그대로 이식
    (session.action() 호출만 generate_vicuna()로 교체)."""
    db = entry["table"]["table_name"]
    init_sql, init_data = build_init_sql(entry)
    container.execute(init_sql, data=init_data)

    history = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Ok."},
    ]
    prompt = entry["description"] + "\n" + entry["add_description"]
    history.append({"role": "user", "content": prompt})

    res = generate_vicuna(tokenizer, model, history)
    history.append({"role": "assistant", "content": res})

    answer, error, finish_reason = "", "", "completed"
    try:
        action = re.search(r"Action: (.*?)\n", res)
        rounds = 0
        while action and action.group(1) == "Operation" and rounds < MAX_ROUND:
            sql_match = re.search(r"```sql\n([\s\S]*?)\n```", res)
            if not sql_match:
                finish_reason = "agent validation failed"
                break
            sql = sql_match.group(1).strip().replace("\n", " ")
            response = container.execute(sql, db)
            history.append({"role": "user", "content": response if response else ""})
            res = generate_vicuna(tokenizer, model, history)
            history.append({"role": "assistant", "content": res})
            action = re.search(r"Action: (.*?)\n", res)
            rounds += 1
        else:
            m = re.search(r"\nFinal Answer:(.*)", res)
            if m:
                answer = m.group(1)
            else:
                answer = ""
                finish_reason = "agent validation failed"
            if rounds >= MAX_ROUND and not answer:
                finish_reason = "task limit reached"
    except Exception as e:
        error = str(e)
        answer = ""
        finish_reason = "unknown"

    if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
        columns = ",".join(f"`{c['name']}`" for c in entry["table"]["table_info"]["columns"])
        md5_query = (
            f"select md5(group_concat(rowhash order by rowhash)) as hash "
            f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
        )
        answer = container.execute(md5_query, db)

    container.execute(f"drop database `{db}`")

    return {
        "status": finish_reason,
        "result": {"answer": str(answer), "type": entry["type"][0], "error": error},
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처음 N문제만 실행 (기본: 전체 300문제)")
    args = parser.parse_args()

    setup_mysql()
    tokenizer, model = load_model()

    entries = [json.loads(l) for l in open(DATA_FILE)]
    if args.limit is not None:
        entries = entries[: args.limit]

    done_indices = set()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                done_indices.add(json.loads(line)["index"])
    print(f"전체 {len(entries)}문제 중 이미 완료된 {len(done_indices)}개는 건너뜁니다")

    container = Container()
    start_time = time.time()

    with open(OUT_FILE, "a") as out:
        for index, entry in enumerate(entries):
            if index in done_indices:
                continue
            t0 = time.time()
            output = run_sample(entry, container, tokenizer, model)
            output["index"] = index
            elapsed = time.time() - t0
            record = {
                "index": index,
                "error": None,
                "info": None,
                "output": output,
                "time": {"timestamp": int(time.time() * 1000), "str": time.strftime("%Y-%m-%d %H:%M:%S")},
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            total_elapsed = time.time() - start_time
            print(f"[{index + 1}/{len(entries)}] status={output['status']:<24} "
                  f"{elapsed:5.1f}s (누적 {total_elapsed / 60:5.1f}분)", flush=True)

    print(f"\n완료: {OUT_FILE}")


if __name__ == "__main__":
    main()
