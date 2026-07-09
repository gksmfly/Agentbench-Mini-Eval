# Agentbench-Mini-Eval

SKT FLY AI 9기 논문 발표 프로젝트. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) (ICLR 2024)의
8개 평가 환경 중 두 환경을 나눠서 재현했습니다.

## 논문 요약
LLM을 정적 문답이 아니라 "에이전트"로서 평가하기 위해, 코드형(OS/DB/지식그래프)·게임형(카드게임/퍼즐/집안일)·
웹형(쇼핑/브라우징) 총 8개 실제 상호작용 환경에서 29개 LLM을 벤치마킹한 논문(ICLR 2024). 상용 API 모델이
오픈소스 모델을 전반적으로 앞선다는 것과, 모델이 약할수록 정해진 출력 형식(Action: ... 같은 프로토콜)을
못 지켜 실패하는 비율이 높다는 것이 핵심 발견이다. 이 저장소는 그중 **Database(DB)**와
**Lateral Thinking Puzzles(LTP)** 두 환경만 재현했다. 자세한 요약은 [`docs/paper_card.md`](docs/paper_card.md) 참고.

- **[`project_db/`](project_db/)** — Database(DB) 환경, 모델 4개(`gpt-3.5-turbo`, `gpt-4o`, `llama-3.1-8b`, `claude-sonnet-5`) 비교
- **[`project_ltp/`](project_ltp/)** — LTP(Lateral Thinking Puzzles) 환경, 모델 5개(`gpt-3.5-turbo`, `gpt-4o`, `claude-sonnet-5`, `vicuna-13b-local`, `llama-3.1-8b`) 비교
- **[`docs/paper_card.md`](docs/paper_card.md)** / **[`docs/project_canvas.md`](docs/project_canvas.md)** — 두 환경 공통 문서 (논문 요약·재현 결과·한계 / 팀 프로젝트 연결)

> `AgentBench/`는 [THUDM/AgentBench](https://github.com/THUDM/AgentBench) 원본 클론이며, 이 저장소에는 커밋되지 않습니다 (`.gitignore` 참고).

---

### 구조

```
project_db/
├── notebooks/
│   ├── demo.ipynb                     발표용 노트북 (Colab+GPU 대응, 4모델 중 택1 라이브 데모 + 결과 시각화)
│   ├── vicuna_db_format_check.ipynb   [탐색] vicuna-13b가 DB 환경 출력 포맷(Action: Operation)을 지키는지 3문제로 빠르게 확인
│   └── vicuna_full_run.ipynb          [탐색] vicuna-13b를 dbbench-std 300문제 전체로 실행 (Docker 없이 Colab에 MySQL 직접 설치, 채점 로직은 원본 그대로 이식) — 진행 중, 아직 공식 비교표에는 미포함
├── src/
│   ├── baseline.py                    프레임워크 없이 LLM 1회 호출 (4개 모델 모두 지원)
│   ├── evaluate.py                    원본 채점 로직을 독립 재구현, 4개 모델 runs.jsonl → metrics.csv
│   └── analyze.py                     비용/토큰/처리속도/SQL실행에러율/응답길이 → analysis.csv
├── data/
│   ├── sample_cases.jsonl             20개 toy 사례 (라이브 데모용)
│   ├── dbbench_standard_gold.jsonl    300문제 정답 라벨 (evaluate.py 채점용)
│   └── dbbench_standard_full.jsonl    300문제 전체(스키마+실제 테이블 데이터) — vicuna_full_run.ipynb 전용
└── results/
    ├── metrics.csv                    4개 모델 통합 집계 지표 (정확도/완료율)
    ├── analysis.csv                   4개 모델 비용/속도/에러율/응답길이
    └── raw/                           모델별 300문제 본실험 원본 로그
```

(논문 요약·발견한 이슈는 루트 [`docs/paper_card.md`](docs/paper_card.md) 참고)

---

## project_db — Database 환경 재현

`dbbench-std` 300문제, 모델 4개 비교. 논문 요약과 발견한 이슈는 [`docs/paper_card.md`](docs/paper_card.md) 참고.

| 모델 | 대표하는 코너 | SELECT | INSERT | UPDATE | Success Rate | 완료율 |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 26% | 37% | 71% | **44.67%** | 93.7% |
| `gpt-4o` | 최고신뢰 후보 | 32% | 32% | 78% | **47.33%** | 53.3%* |
| `llama-3.1-8b` (로컬 GPU) | 저비용 오픈소스 후보 | 0% | 0% | 0% | **0%** | 0%† |
| `claude-sonnet-5` | 타 벤더 검증 후보‡ | 62% | 54% | 88% | **68.00%** | 97.0% |

\* gpt-4o는 정확도는 최고지만 프레임워크의 "context limit" 오분류(25.7%)로 완료율이 낮게 잡힘
† llama-3.1-8b는 포맷 미준수(82.3%)로 즉시 실패 — 자세한 원인은 `docs/paper_card.md` "한계" 참고
‡ claude-sonnet-5는 정확도·완료율 모두 압도적으로 높지만, gpt-4o(2024년 모델)보다 2년 뒤에 나온
  모델이라 이 우위가 "벤더 차이"인지 "세대 차이"인지 구분할 수 없음 — 자세한 내용은 `docs/paper_card.md` "한계" 참고

**비용·속도·에러율 추가 분석** (`project_db/src/analyze.py`, `project_db/results/analysis.csv`):

| 모델 | 300문제 비용 | 정답당 비용 | SQL 실행 에러율 | 평균 응답 길이 |
|---|---|---|---|---|
| `gpt-3.5-turbo` | $0.44 | **$0.0033** | **50.7%** | 134자 |
| `gpt-4o` | $1.61 | $0.0113 | 7.3% | **259자** |
| `llama-3.1-8b` | $0.01 (실제로는 로컬 GPU라 $0) | — | 14.0% | 78자 |
| `claude-sonnet-5` | $2.18 | $0.0107 | 7.3% | 212자 |

gpt-3.5-turbo는 SQL 문법 오류를 훨씬 많이 내면서도(50.7%) 재시도로 복구해 최종 정확도는 gpt-4o와
비슷 — "정답당 비용"으로 보면 gpt-4o보다 3.4배 저렴. claude-sonnet-5는 정답당 비용이 gpt-4o와
거의 같으면서(약 $0.0107 vs $0.0113) Success Rate는 20%p 이상 높음. 자세한 내용은 `docs/paper_card.md` 참고.

### 빠른 실행

`baseline.py`와 `evaluate.py`는 `project_db/` 폴더만으로 독립 실행됩니다 (AgentBench 원본 클론 불필요).

```bash
cd project_db
pip install requests pandas matplotlib transformers accelerate bitsandbytes

# 1) 핵심 아이디어 최소 재현 — LLM 1회 호출로 SQL 제안 확인 (4개 모델 다 지원)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...      # claude-sonnet-5 데모용
export HF_TOKEN=hf_...                   # llama-3.1-8b 데모용 (게이트 모델, huggingface.co 라이선스 동의 필요)
python src/baseline.py --index 0 --model gpt-3.5-turbo
python src/baseline.py --index 0 --model gpt-4o
python src/baseline.py --index 0 --model claude-sonnet-5
python src/baseline.py --index 0 --model llama-3.1-8b   # 로컬 GPU에 4비트 양자화로 직접 로드 (API 호출 아님)

# 2) 실제 300문제 본실험 로그로부터 4개 모델 지표 재계산 (원본 채점 로직과 결과 일치 확인됨)
python src/evaluate.py

# 2-1) 비용/속도/에러율 등 추가 분석
pip install tiktoken
python src/analyze.py

# 3) 발표용 노트북 (Colab 권장: 런타임 유형을 GPU로 설정)
jupyter notebook notebooks/demo.ipynb
```

### 300문제 본실험을 처음부터 다시 돌리려면

`results/`에 이미 4개 모델 원본 로그가 포함되어 있어 위 "빠른 실행"만으로 충분하지만,
직접 처음부터 재현하려면 아래 과정이 필요합니다 (Docker + 실제 MySQL 필요, `project_db/`가 아니라
별도로 클론한 [THUDM/AgentBench](https://github.com/THUDM/AgentBench) `v0.2` 위에서 진행):

1. `git clone https://github.com/THUDM/AgentBench.git && cd AgentBench && git checkout v0.2`
2. `conda create -n agent-bench python=3.9 && conda activate agent-bench && pip install -r requirements.txt`
3. Docker Desktop 실행, `docker pull mysql:8.0` (⚠️ `mysql:latest`가 아닌 `8.0` 고정 — 아래 "발견한 이슈" 참고)
4. `configs/agents/api_agents.yaml`에 모델 4개 등록 (`gpt-3.5-turbo`, `gpt-4o`는 `openai-chat.yaml` import,
   `llama-3.1-8b`는 로컬 GPU에서 4비트 양자화로 직접 서빙한 뒤 그 OpenAI 호환 로컬 엔드포인트
   (`http://localhost:8000/v1/chat/completions`)를 가리키는 `local-llama-chat.yaml`을 새로 만들어 import —
   응답 형식이 OpenAI 호환이라 `openai-chat.yaml`과 구조 동일,
   `claude-sonnet-5`는 Anthropic Messages API(`https://api.anthropic.com/v1/messages`)를 쓰는
   `claude-chat.yaml`을 새로 만들어 import — 요청/응답 형식이 달라 `body.thinking: disabled`와
   `return_format: "{response[content][0][text]}"`을 별도로 지정)
5. `python -m src.start_task -a` (Task Server 기동, 포트 5000~5010)
6. 새 터미널에서 모델별로 `python -m src.assigner --config configs/assignments/db_only_<model>.yaml`
   (`os-std` 제외하고 `dbbench-std`만 도는 assignment config, 모델 이름만 바꿔 4번 실행)
7. `outputs/<timestamp>/<model>/dbbench-std/{runs.jsonl,overall.json}` 확인 후 `project_db/results/raw/`로 복사

### 발견한 이슈 5가지

**1) MySQL 버전 드리프트 — 채점 자체가 실패하던 버그**
`Interaction.py`가 MySQL 이미지를 버전 고정 없이 `mysql`(=`latest`)로 띄워, 논문 작성 시점(2023, MySQL 8.0)과
지금(2026, MySQL 9.x) 사이 `MD5()` 함수 처리 방식이 바뀌면서 INSERT/UPDATE 채점 쿼리가
`FUNCTION <db>.MD5 does not exist` 에러로 실패. `mysql:8.0` 고정으로 해결 (INSERT 0%→37%, UPDATE 0%→71%).

**2) gpt-4o의 "컨텍스트 한도 초과" 오분류**
완료율 53.3%, 25.7%가 "agent context limit"으로 분류됨. `gpt-4o`는 128k 토큰급이라 실제 컨텍스트 초과
가능성은 낮음. `http_agent.py`의 `check_context_limit()`이 HTTP 에러 응답 텍스트의 키워드 매칭만으로
판단하는 허술한 로직이라, 레이트리밋 등 다른 에러까지 오분류할 수 있음을 코드로 확인. 정확한 원인은
로그에 안 남아 특정하지 못함 (프레임워크의 한계로 기록).

**3) llama-3.1-8b의 포맷 미준수**
완료율 0%, 82.3%가 포맷 미준수로 실패. 지시된 고정 문구 `Action: Operation` 대신 `Action: Update`처럼
SQL 종류에 맞춰 말을 바꿔 써서 정규식 매칭에 실패. 논문의 "약한 모델일수록 지시사항을 못 따른다"는
핵심 주장을 2026년 소형 오픈소스 모델로 그대로 재현.

**4) 정확도가 다가 아니다**
`gpt-3.5-turbo`는 SQL 실행 에러율이 50.7%로 셋 중 가장 높은데도 재시도로 복구해, 최종 Success
Rate(44.67%)는 `gpt-4o`(47.33%)와 비슷합니다. 반면 정답 하나당 비용은 `gpt-4o`가 `gpt-3.5-turbo`의
3.4배($0.0113 vs $0.0033) — 정확도 차이는 3%p뿐인데 비용은 훨씬 큽니다.

**5) 타 벤더 모델(claude-sonnet-5)이 압도적으로 우수 — 다만 벤더 차이인지 세대 차이인지는 불명확**
`claude-sonnet-5`는 Success Rate 68.00%, 완료율 97.0%로 4개 모델 중 모든 지표에서 최고입니다.
SQL 실행 에러율(7.3%)도 gpt-4o와 동일하게 낮고, 정답당 비용(약 $0.0107)도 gpt-4o(약 $0.0113)와
거의 같습니다 — 즉 "같은 비용에 훨씬 정확"합니다. 그런데 gpt-4o는 2024년 모델, claude-sonnet-5는
2026년 모델이라 두 모델 사이에 2년의 세대 차이가 있습니다. 이 우위가 "Anthropic이 이 태스크를
더 잘한다"는 벤더 차이 때문인지, 단순히 "더 최신 모델이라 더 잘한다"는 세대 차이 때문인지
이 실험만으로는 구분할 수 없습니다 — 동시대에 출시된 모델끼리 비교해야 공정한 벤더 비교가 됩니다.

다섯 이슈 모두 채점 쿼리/로직/프롬프트는 건드리지 않고 인프라 버전 고정 또는 원인 분석만 했으므로,
재현의 타당성을 해치지 않습니다. 자세한 근거는 `docs/paper_card.md` 참고.

---

## project_ltp — LTP 환경 재현

공식 AgentBench v0.2로 Lateral Thinking Puzzles(LTP) 환경을 실행한 결과. 실행은 CLI(서버+Docker)로
수행했고, 노트북은 측정 결과 시각화만 담당. dev 서브셋(20문제) 기준, 채점 LLM은 다섯 실험 모두
`gpt-3.5-turbo`로 고정(공정 비교).

| 모델 | 역할 | 게임 수 | Game Progress | Single Game Accuracy | 완료 / 라운드초과 |
|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 20 | **5.8%** | 20.6% | 5 / 15 |
| `gpt-4o` | 신뢰성 상한선 | 13* | 5.1%* | **38.5%** | 0 / 13 |
| `claude-sonnet-5` | 타 벤더 검증 후보 | 20 | 3.8% | 17.2% | **13 / 7** |
| `vicuna-13b-local` | 로컬 오픈소스 비교군 | 20 | 3.3% | 7.4% | 0 / 20 |
| `llama-3.1-8b` | 저비용 오픈소스 하한선 | 20 | **0.0%** | 6.2% | 0 / 20 |

\* gpt-4o는 실행 중 서버 연결이 끊겨(NETWORK_ERROR) 재시도 — 두 시도를 인덱스 기준으로 병합해
n=13(0~6번 문제 누락)으로 집계. 다른 모델(n=20)보다 표본이 작아 순위 해석에 주의 필요.

지표(논문 Appendix F): GP(Game Progress)가 메인 지표, SGA(Single Game Accuracy)는 보조 지표.
전 모델이 GP 6% 미만으로 대부분의 게임에서 라운드 초과(task limit reached)로 종료 — LTP는 DB보다
훨씬 긴 멀티턴 추론을 요구하는 환경임을 보여줌. claude-sonnet-5만 완료율이 확연히 높은데(65%),
DB 환경에서 claude-sonnet-5의 완료율(97%)이 압도적이었던 것과 같은 패턴 — 다만 완료율 우위가
GP 우위로 이어지지는 않음.

### 구조

```
project_ltp/
├── notebooks/
│   └── demo.ipynb                     발표용 노트북 (라이브 데모 + 결과 시각화 + 채점 로직 검증, project_db와 동일 패턴)
├── src/
│   ├── baseline.py                    프레임워크 없이 LLM 1회 호출로 호스트 역할 최소 재현 (4개 모델 지원)
│   ├── evaluate.py                    원본 채점 로직(GP/SGA/RE/QR)을 독립 재구현, raw 로그와 대조 검증 → metrics.csv
│   └── analyze.py                     비용/토큰/처리속도/라운드사용량/응답길이 → analysis.csv
├── data/
│   ├── ltp_dev_gold.jsonl             20개 dev 퍼즐의 정답 키포인트 개수 (evaluate.py 채점용)
│   └── sample_cases.jsonl             8개 toy 사례 (baseline.py 데모용)
└── results/
    ├── metrics.csv                    5개 모델 통합 집계 지표 (project_db와 동일한 long format)
    ├── analysis.csv                   5개 모델 비용/속도/라운드사용량/응답길이
    └── raw/                           runs.jsonl 원본 로그 (모델별)
```
