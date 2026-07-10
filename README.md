# Agentbench-Mini-Eval

SKT FLY AI 9기 논문 발표 프로젝트. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) (ICLR 2024)의
8개 평가 환경 중 두 환경을 나눠서 재현했습니다.

## 1 논문 요약
LLM을 정적 문답이 아니라 "에이전트"로서 평가하기 위해, 코드형(OS/DB/지식그래프)·게임형(카드게임/퍼즐/집안일)·
웹형(쇼핑/브라우징) 총 8개 실제 상호작용 환경에서 29개 LLM을 벤치마킹한 논문(ICLR 2024). 상용 API 모델이
오픈소스 모델을 전반적으로 앞선다는 것과, 모델이 약할수록 정해진 출력 형식(Action: ... 같은 프로토콜)을
못 지켜 실패하는 비율이 높다는 것이 핵심 발견이다. 이 저장소는 그중 **Database**와
**Lateral Thinking Puzzles(LTP)** 두 환경만 재현했다. 자세한 요약은 [`docs/paper_card.md`](docs/paper_card.md) 참고.

- **[`project_db/`](project_db/)** — Database(DB) 환경, 모델 5개(`gpt-3.5-turbo`, `gpt-4o`, `llama-3.1-8b`, `claude-sonnet-5`, `vicuna-13b-local`) 비교
- **[`project_ltp/`](project_ltp/)** — LTP(Lateral Thinking Puzzles) 환경, 모델 5개(`gpt-3.5-turbo`, `gpt-4o`, `claude-sonnet-5`, `vicuna-13b-local`, `llama-3.1-8b`) 비교
- **[`docs/paper_card.md`](docs/paper_card.md)** / **[`docs/project_canvas.md`](docs/project_canvas.md)** — 두 환경 공통 문서 (논문 요약·재현 결과·한계 / 팀 프로젝트 연결)

> `AgentBench/`는 [THUDM/AgentBench](https://github.com/THUDM/AgentBench) 원본 클론이며, 이 저장소에는 커밋되지 않습니다 (`.gitignore` 참고).

---

### 2. 구조

```
project_db/
├── notebooks/
│   └── demo.ipynb                     발표용 노트북 (Colab+GPU 대응, 5모델 중 택1 라이브 데모 + 결과 시각화)
├── src/
│   ├── baseline.py                    프레임워크 없이 LLM 1회 호출 (5개 모델 모두 지원)
│   ├── evaluate.py                    원본 채점 로직을 독립 재구현, 5개 모델 runs.jsonl → metrics.csv
│   ├── analyze.py                     비용/토큰/처리속도/SQL실행에러율/응답길이 → analysis.csv
│   └── run_vicuna_full.py             [탐색] vicuna-13b를 dbbench-std 300문제 전체로 원격 GPU 서버에서 실행 (MySQL 설치부터 채점까지 이 파일 하나로 처리) — 진행 중, 아직 공식 비교표에는 미포함
├── data/
│   ├── sample_cases.jsonl             20개 toy 사례 (라이브 데모용)
│   ├── dbbench_standard_gold.jsonl    300문제 정답 라벨 (evaluate.py 채점용)
│   └── dbbench_standard_full.jsonl    300문제 전체(스키마+실제 테이블 데이터) — run_vicuna_full.py 전용
└── results/
    ├── metrics.csv                    5개 모델 통합 집계 지표 (정확도/완료율)
    ├── analysis.csv                   5개 모델 비용/속도/에러율/응답길이
    └── raw/                           모델별 300문제 본실험 원본 로그
```

(논문 요약·발견한 이슈는 루트 [`docs/paper_card.md`](docs/paper_card.md) 참고)

---

## 3 project_db — Database 환경 재현

`dbbench-std` 300문제, 모델 5개 비교. 논문 요약과 발견한 이슈는 [`docs/paper_card.md`](docs/paper_card.md) 참고.

| 모델 | 대표하는 코너 | SELECT | INSERT | UPDATE | Success Rate | 완료율 |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 26% | 37% | 71% | **44.67%** | 93.7% |
| `gpt-4o` | 최고신뢰 후보 | 32% | 32% | 78% | **47.33%** | 53.3%* |
| `claude-sonnet-5` | 타 벤더 검증 후보 | 62% | 54% | 88% | **68.00%** | 97.0% |
| `llama-3.1-8b` | 저비용 오픈소스 후보 | 0% | 0% | 0% | **0%** | 0%† |
| `vicuna-13b-local` | 구형 오픈소스 검증 | 2% | 0% | 0% | **0.67%** | 22.7% |

**비용·속도·에러율 추가 분석** (`project_db/src/analyze.py`, `project_db/results/analysis.csv`):

| 모델 | 300문제 비용 | 정답당 비용 | SQL 실행 에러율 | 평균 응답 길이 |
|---|---|---|---|---|
| `gpt-3.5-turbo` | $0.44 | **$0.0033** | 51.7% | 134자 |
| `gpt-4o` | $1.61 | $0.0113 | 7.3% | 259자 |
| `claude-sonnet-5` | $2.18 | $0.0107 | 7.3% | 212자 |
| `llama-3.1-8b` | $0 (로컬 GPU) | — | 14.3% | 78자 |
| `vicuna-13b-local` | $0 (로컬 GPU) | $0.00 | **64.3%** | **644자** |

### 3.1 빠른 실행

```bash
cd project_db
pip install requests pandas matplotlib transformers accelerate bitsandbytes

# 1) 핵심 아이디어 최소 재현 — LLM 1회 호출로 SQL 제안 확인 (5개 모델 다 지원)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...      # claude-sonnet-5 데모용
python src/baseline.py --index 0 --model gpt-3.5-turbo
python src/baseline.py --index 0 --model gpt-4o
python src/baseline.py --index 0 --model claude-sonnet-5
python src/baseline.py --index 0 --model llama-3.1-8b   # 로컬 GPU에 4비트 양자화로 직접 로드 (API 호출 아님)
python src/baseline.py --index 0 --model vicuna-13b-local   # 로컬 GPU에 4비트 양자화로 직접 로드 (API 호출 아님)

# 2) 실제 300문제 본실험 로그로부터 5개 모델 지표 재계산 (원본 채점 로직과 결과 일치 확인됨)
python src/evaluate.py

# 2-1) 비용/속도/에러율 등 추가 분석
pip install tiktoken
python src/analyze.py

# 3) 발표용 노트북 (Colab 권장: 런타임 유형을 GPU로 설정)
jupyter notebook notebooks/demo.ipynb
```

## 4 project_ltp — LTP 환경 재현

공식 AgentBench v0.2로 Lateral Thinking Puzzles(LTP) 환경을 실행한 결과. 실행은 CLI(서버+Docker)로
수행했고, 노트북은 측정 결과 시각화만 담당. dev 서브셋(20문제) 기준, 채점 LLM은 다섯 실험 모두
`gpt-3.5-turbo`로 고정(공정 비교).

| 모델 | 역할 | 게임 수 | Game Progress | Single Game Accuracy | 완료 / 라운드초과 |
|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 20 | 5.8% | 20.6% | 5 / 15 |
| `gpt-4o` | 신뢰성 상한선 | 20 | **7.9%** | **43.2%** | 1 / 19 |
| `claude-sonnet-5` | 타 벤더 검증 후보 | 20 | 3.8% | 17.2% | **13 / 7** |
| `vicuna-13b-local` | 로컬 오픈소스 비교군 | 20 | 3.3% | 7.4% | 0 / 20 |
| `llama-3.1-8b` | 저비용 오픈소스 하한선 | 20 | 0.0% | 6.2% | 0 / 20 |

**비용·속도·에러율 추가 분석** (`project_ltp/src/analyze.py`, `project_ltp/results/analysis.csv`):

| 모델 | dev 20문제 비용 | 처리 속도 | 평균 라운드 사용 | Host Irrelevant 응답률 | 평균 응답 길이 |
|---|---|---|---|---|---|
| `gpt-3.5-turbo` | $0.1529 | 0.64 게임/분 | 24.15 / 25 | 38.6% | 102자 |
| `gpt-4o` | $0.7968 | 0.59 게임/분 | 24.2 / 25 | 32.3% | 93자 |
| `claude-sonnet-5` | $0.5895 | 0.61 게임/분 | 25.0 / 25 | 38.4% | **146자** |
| `llama-3.1-8b` | $0 (로컬 GPU) | **1.03 게임/분** | 25.0 / 25 | **10.4%** | 139자 |
| `vicuna-13b-local` | $0 (로컬 GPU) | 0.87 게임/분 | 25.0 / 25 | 30.0% | 129자 |

### 4.1 빠른 실행

```bash
cd project_ltp
pip install requests pandas matplotlib transformers accelerate bitsandbytes

# 1) 핵심 아이디어 최소 재현 — LLM 1회 호출로 호스트 역할(Yes/No/Irrelevant 판정) 확인
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...      # claude-sonnet-5 데모용
python src/baseline.py --index 0 --model gpt-3.5-turbo
python src/baseline.py --index 0 --model gpt-4o
python src/baseline.py --index 0 --model claude-sonnet-5
python src/baseline.py --index 0 --model llama-3.1-8b   # 로컬 GPU에 4비트 양자화로 직접 로드 (API 호출 아님)
python src/baseline.py --index 0 --model vicuna-13b-local   # 로컬 GPU에 4비트 양자화로 직접 로드 (API 호출 아님)

# 2) 실제 dev 서브셋(20문제) 본실험 로그로부터 5개 모델 지표 재계산 (원본 채점 로직과 결과 일치 확인됨)
python src/evaluate.py

# 2-1) 비용/속도/라운드사용량 등 추가 분석
pip install tiktoken
python src/analyze.py

# 3) 발표용 노트북 (Colab 권장: 런타임 유형을 GPU로 설정, vicuna-13b-local 로컬 로드용)
jupyter notebook notebooks/demo.ipynb
```
