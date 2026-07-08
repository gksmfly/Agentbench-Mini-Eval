# AgentBench — Database 환경 재현 (모델 3개 비교)

SKT FLY AI 9기 논문 발표용 프로젝트. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) (ICLR 2024)의
8개 평가 환경 중 **Database(DB) 환경**만 재현하고, 3개 모델(최저비용/최고신뢰/저비용오픈소스)을 비교했습니다.
논문 요약과 발견한 이슈는 [`docs/paper_card.md`](docs/paper_card.md) 참고. 팀 프로젝트 연결은
[`docs/project_canvas.md`](docs/project_canvas.md)에 별도 팀원이 작성 예정.

## 결과 요약

`dbbench-std` 300문제, 모델 3개 비교:

| 모델 | 대표하는 코너 | SELECT | INSERT | UPDATE | Success Rate | 완료율 |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 26% | 37% | 71% | **44.67%** | 93.7% |
| `gpt-4o` | 최고신뢰 후보 | 32% | 32% | 78% | **47.33%** | 53.3%* |
| `llama-3.1-8b` (Groq, 무료) | 저비용 오픈소스 후보 | 0% | 0% | 0% | **0%** | 0%† |

\* gpt-4o는 정확도는 최고지만 프레임워크의 "context limit" 오분류(25.7%)로 완료율이 낮게 잡힘
† llama-3.1-8b는 포맷 미준수(82.3%)로 즉시 실패 — 자세한 원인은 `docs/paper_card.md` "한계" 참고

**비용·속도·에러율 추가 분석** (`src/analyze.py`, `results/analysis.csv`):

| 모델 | 300문제 비용 | 정답당 비용 | SQL 실행 에러율 | 평균 응답 길이 |
|---|---|---|---|---|
| `gpt-3.5-turbo` | $0.44 | **$0.0033** | **50.7%** | 134자 |
| `gpt-4o` | $1.61 | $0.0113 | 7.3% | **259자** |
| `llama-3.1-8b` | $0.01 (실제로는 Groq 무료) | — | 14.0% | 78자 |

gpt-3.5-turbo는 SQL 문법 오류를 훨씬 많이 내면서도(50.7%) 재시도로 복구해 최종 정확도는 gpt-4o와
비슷 — "정답당 비용"으로 보면 gpt-4o보다 3.4배 저렴. 자세한 내용은 `docs/paper_card.md` 참고.

전체 로그: [`results/raw/`](results/raw/) · 집계: [`results/metrics.csv`](results/metrics.csv), [`results/analysis.csv`](results/analysis.csv)

## 구조

```
project/
├── README.md                          이 파일
├── notebooks/demo.ipynb               발표용 노트북 (Colab+GPU 대응, 3모델 라이브 데모 + 결과 시각화)
├── src/
│   ├── baseline.py                    프레임워크 없이 LLM 1회 호출 (3개 모델 모두 지원)
│   ├── evaluate.py                    원본 채점 로직을 독립 재구현, 3개 모델 runs.jsonl → metrics.csv
│   └── analyze.py                     비용/토큰/처리속도/SQL실행에러율/응답길이 → analysis.csv
├── data/
│   ├── sample_cases.jsonl             20개 toy 사례 (라이브 데모용)
│   └── dbbench_standard_gold.jsonl    300문제 정답 라벨 (evaluate.py 채점용)
├── results/
│   ├── metrics.csv                    3개 모델 통합 집계 지표 (정확도/완료율)
│   ├── analysis.csv                   3개 모델 비용/속도/에러율/응답길이
│   └── raw/                           모델별 300문제 본실험 원본 로그
└── docs/
    ├── paper_card.md                  논문 요약, 3모델 비교, 발견한 이슈 3가지
    └── project_canvas.md              팀 프로젝트 연결 (담당 팀원 작성 예정)
```

## 빠른 실행

`baseline.py`와 `evaluate.py`는 이 폴더만으로 독립 실행됩니다 (AgentBench 원본 클론 불필요).

```bash
pip install requests pandas matplotlib

# 1) 핵심 아이디어 최소 재현 — LLM 1회 호출로 SQL 제안 확인 (3개 모델 다 지원)
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...              # llama-3.1-8b 데모용 (무료 발급)
python src/baseline.py --index 0 --model gpt-3.5-turbo
python src/baseline.py --index 0 --model gpt-4o
python src/baseline.py --index 0 --model llama-3.1-8b-instant

# 2) 실제 300문제 본실험 로그로부터 3개 모델 지표 재계산 (원본 채점 로직과 결과 일치 확인됨)
python src/evaluate.py

# 2-1) 비용/속도/에러율 등 추가 분석
pip install tiktoken
python src/analyze.py

# 3) 발표용 노트북 (Colab 권장: 런타임 유형을 GPU로 설정)
jupyter notebook notebooks/demo.ipynb
```

## 300문제 본실험을 처음부터 다시 돌리려면

`results/`에 이미 3개 모델 원본 로그가 포함되어 있어 위 "빠른 실행"만으로 충분하지만,
직접 처음부터 재현하려면 아래 과정이 필요합니다 (Docker + 실제 MySQL 필요, `project/`가 아니라
별도로 클론한 [THUDM/AgentBench](https://github.com/THUDM/AgentBench) `v0.2` 위에서 진행):

1. `git clone https://github.com/THUDM/AgentBench.git && cd AgentBench && git checkout v0.2`
2. `conda create -n agent-bench python=3.9 && conda activate agent-bench && pip install -r requirements.txt`
3. Docker Desktop 실행, `docker pull mysql:8.0` (⚠️ `mysql:latest`가 아닌 `8.0` 고정 — 아래 "발견한 이슈" 참고)
4. `configs/agents/api_agents.yaml`에 모델 3개 등록 (`gpt-3.5-turbo`, `gpt-4o`는 `openai-chat.yaml` import,
   `llama-3.1-8b`는 Groq 엔드포인트(`https://api.groq.com/openai/v1/chat/completions`)를 쓰는
   `groq-chat.yaml`을 새로 만들어 import — OpenAI 호환 포맷이라 `openai-chat.yaml`과 구조 동일)
5. `python -m src.start_task -a` (Task Server 기동, 포트 5000~5010)
6. 새 터미널에서 모델별로 `python -m src.assigner --config configs/assignments/db_only_<model>.yaml`
   (`os-std` 제외하고 `dbbench-std`만 도는 assignment config, 모델 이름만 바꿔 3번 실행)
7. `outputs/<timestamp>/<model>/dbbench-std/{runs.jsonl,overall.json}` 확인 후 `project/results/raw/`로 복사

## 발견한 이슈 4가지

### 1) MySQL 버전 드리프트 — 채점 자체가 실패하던 버그
`Interaction.py`가 MySQL 이미지를 버전 고정 없이 `mysql`(=`latest`)로 띄워, 논문 작성 시점(2023, MySQL 8.0)과
지금(2026, MySQL 9.x) 사이 `MD5()` 함수 처리 방식이 바뀌면서 INSERT/UPDATE 채점 쿼리가
`FUNCTION <db>.MD5 does not exist` 에러로 실패. `mysql:8.0` 고정으로 해결 (INSERT 0%→37%, UPDATE 0%→71%).

### 2) gpt-4o의 "컨텍스트 한도 초과" 오분류
완료율 53.3%, 25.7%가 "agent context limit"으로 분류됨. `gpt-4o`는 128k 토큰급이라 실제 컨텍스트 초과
가능성은 낮음. `http_agent.py`의 `check_context_limit()`이 HTTP 에러 응답 텍스트의 키워드 매칭만으로
판단하는 허술한 로직이라, 레이트리밋 등 다른 에러까지 오분류할 수 있음을 코드로 확인. 정확한 원인은
로그에 안 남아 특정하지 못함 (프레임워크의 한계로 기록).

### 3) llama-3.1-8b의 포맷 미준수
완료율 0%, 82.3%가 포맷 미준수로 실패. 지시된 고정 문구 `Action: Operation` 대신 `Action: Update`처럼
SQL 종류에 맞춰 말을 바꿔 써서 정규식 매칭에 실패. 논문의 "약한 모델일수록 지시사항을 못 따른다"는
핵심 주장을 2026년 소형 오픈소스 모델로 그대로 재현.

### 4) 정확도가 다가 아니다
`gpt-3.5-turbo`는 SQL 실행 에러율이 50.7%로 셋 중 가장 높은데도 재시도로 복구해, 최종 Success
Rate(44.67%)는 `gpt-4o`(47.33%)와 비슷합니다. 반면 정답 하나당 비용은 `gpt-4o`가 `gpt-3.5-turbo`의
3.4배($0.0113 vs $0.0033) — 정확도 차이는 3%p뿐인데 비용은 훨씬 큽니다.

네 이슈 모두 채점 쿼리/로직/프롬프트는 건드리지 않고 인프라 버전 고정 또는 원인 분석만 했으므로,
재현의 타당성을 해치지 않습니다. 자세한 근거는 `docs/paper_card.md` 참고.
