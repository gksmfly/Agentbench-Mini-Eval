# AgentBench — Database 환경 재현

SKT FLY AI 9기 논문 발표용 프로젝트. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) (ICLR 2024)의
8개 평가 환경 중 **Database(DB) 환경**만 재현했습니다. 논문 요약과 한계는 [`docs/paper_card.md`](docs/paper_card.md) 참고.

## 결과 요약

`gpt-3.5-turbo` 기준, `dbbench-std` 300문제:

| 지표 | 값 |
|---|---|
| SELECT 계열 정확도 | 26% |
| INSERT 정확도 | 37% |
| UPDATE 정확도 | 71% |
| **Success Rate** (3개 평균) | **44.67%** |
| 정상 완료율 | 93.7% |

(논문 Table 3의 `gpt-3.5-turbo` DB 점수: 36.7% — 모델 스냅샷이 정확히 같지 않아 생기는 자연스러운 차이 범위)

전체 로그: [`results/raw/dbbench_std_runs.jsonl`](results/raw/dbbench_std_runs.jsonl) · 집계: [`results/metrics.csv`](results/metrics.csv)

## 구조

```
paper-project/
├── README.md                          이 파일
├── notebooks/demo.ipynb               발표용 실행 노트북 (라이브 데모 + 결과 시각화)
├── src/
│   ├── baseline.py                    프레임워크 없이 LLM 1회 호출로 SQL 제안 (핵심 아이디어 최소 재현)
│   └── evaluate.py                    원본 채점 로직을 독립 재구현, runs.jsonl → metrics.csv
├── data/
│   ├── sample_cases.jsonl             20개 toy 사례 (라이브 데모용)
│   └── dbbench_standard_gold.jsonl    300문제 정답 라벨 (evaluate.py 채점용)
├── results/
│   ├── metrics.csv                    최종 집계 지표
│   └── raw/                           실제 300문제 본실험 원본 로그 (THUDM/AgentBench 프레임워크가 생성)
└── docs/
    ├── paper_card.md                  논문 요약, 한계, 발견한 이슈
    └── project_canvas.md              팀 프로젝트 연결 (추후 작성)
```

## 빠른 실행

`baseline.py`와 `evaluate.py`는 이 폴더만으로 독립 실행됩니다 (AgentBench 원본 클론 불필요).

```bash
pip install requests pandas matplotlib

# 1) 핵심 아이디어 최소 재현 — LLM 1회 호출로 SQL 제안 확인
export OPENAI_API_KEY=sk-...
python src/baseline.py --index 0

# 2) 실제 300문제 본실험 로그로부터 지표 재계산 (원본 채점 로직과 결과 일치 확인됨)
python src/evaluate.py

# 3) 발표용 노트북
jupyter notebook notebooks/demo.ipynb
```

## 300문제 본실험을 처음부터 다시 돌리려면

`results/`에 이미 원본 로그가 포함되어 있어 위 "빠른 실행"만으로 충분하지만,
직접 처음부터 재현하려면 아래 과정이 필요합니다 (Docker + 실제 MySQL 필요, `paper-project/`가 아니라
별도로 클론한 [THUDM/AgentBench](https://github.com/THUDM/AgentBench) `v0.2` 위에서 진행):

1. `git clone https://github.com/THUDM/AgentBench.git && cd AgentBench && git checkout v0.2`
2. `conda create -n agent-bench python=3.9 && conda activate agent-bench && pip install -r requirements.txt`
3. Docker Desktop 실행, `docker pull mysql:8.0` (⚠️ `mysql:latest`가 아닌 `8.0` 고정 — 아래 "발견한 이슈" 참고)
4. `configs/agents/openai-chat.yaml`의 `model`을 살아있는 모델명(예: `gpt-3.5-turbo`)으로, API 키를 실제 키로 설정
5. `python -m src.start_task -a` (Task Server 기동, 포트 5000~5010)
6. 새 터미널에서 `python -m src.assigner --config configs/assignments/db_only.yaml` (아래 참고, os-std 제외하고 DB만 실행)
7. `outputs/<timestamp>/gpt-3.5-turbo-0613/dbbench-std/{runs.jsonl,overall.json}` 확인

`configs/assignments/db_only.yaml`은 기본 `default.yaml`에서 `os-std`를 뺀 버전입니다:
```yaml
import: definition.yaml
concurrency:
  task: { dbbench-std: 5 }
  agent: { gpt-3.5-turbo-0613: 5 }
assignments:
  - agent: [gpt-3.5-turbo-0613]
    task: [dbbench-std]
output: "outputs/{TIMESTAMP}"
```

## 발견한 이슈: MySQL 버전 드리프트

AgentBench의 DB 채점 코드(`src/server/tasks/dbbench/Interaction.py`)는 MySQL 이미지를 버전 고정 없이
`mysql`(=`latest`)로 띄웁니다. 논문 작성 시점(2023)엔 `latest`가 MySQL 8.0이었지만, 지금은 9.x로
넘어가면서 INSERT/UPDATE 채점에 쓰이는 `MD5()` 함수 처리 방식이 바뀌어 **채점 쿼리 자체가
`FUNCTION <db>.MD5 does not exist` 에러로 실패**, INSERT/UPDATE 정확도가 항상 0%로 나오는 문제를 발견했습니다.

`Interaction.py`의 `Container.__init__` 기본 이미지를 `"mysql:8.0"`으로 고정해 해결했고,
그 결과 INSERT 0%→37%, UPDATE 0%→71%로 정상 채점되었습니다. 채점 쿼리/로직/프롬프트는 전혀
건드리지 않고 인프라 버전만 원 실험 환경에 맞춘 것이라, 재현의 타당성을 해치지 않습니다
(자세한 근거는 `docs/paper_card.md`의 "한계" 참고).
