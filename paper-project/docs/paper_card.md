# Paper Card

## 논문 정보
- **제목**: AgentBench: Evaluating LLMs as Agents
- **저자**: Xiao Liu, Hao Yu, Hanchen Zhang, et al. (Tsinghua University, OSU, UC Berkeley)
- **연도 / venue**: ICLR 2024
- **arXiv**: 2308.03688
- **공식 코드**: https://github.com/THUDM/AgentBench (본 프로젝트는 `v0.2` 태그 기준)

## 한 줄 요약
LLM을 "에이전트"로서 평가하기 위해, OS/DB/지식그래프 같은 **코드 기반**, 카드게임/퍼즐/집안일 같은 **게임 기반**, 쇼핑/브라우징 같은 **웹 기반** — 총 8개의 실제 상호작용 환경에서 29개 LLM을 벤치마킹한 논문. 우리는 이 중 **Database(DB) 환경**만 재현했다.

## 문제정의
- 기존 LLM 평가(MMLU, HELM 등)는 단발성 정적 태스크 위주라, "여러 라운드에 걸쳐 환경과 상호작용하며 목표를 완수하는 능력"을 측정하지 못함
- 기존 텍스트 게임/임베디드 에이전트 벤치마크는 특정 환경 하나에만 집중되어 있어, LLM의 다양한 실제 활용 시나리오를 종합적으로 보여주지 못함

## 핵심 아이디어
1. Code / Game / Web 3계열, 8개 환경을 하나의 통일된 (Thought, Action) 형식으로 평가
2. 각 환경을 Docker로 격리하고, Server-Client 구조(Task Server ↔ Agent ↔ Evaluation Client)로 다양한 LLM을 동일 조건에서 비교
3. 환경별 난이도 편차를 보정하기 위해, 각 태스크의 "전체 모델 평균 점수의 역수"를 가중치로 삼아 Overall Score를 계산

## 모델/알고리즘 (DB 환경 기준)
- **입력**: 자연어 질문 + 테이블 스키마/이름 (실제 테이블 내용은 MySQL에 미리 적재, LLM에는 설명만 제공)
- **출력(Action)**: 한 번에 SQL 한 문장, 마크다운 코드블록 형식 강제
- **루프**: LLM이 SQL 제안 → 실제 MySQL에서 실행 → 결과를 다시 LLM에 피드백 → 최대 15라운드 반복 → `Action: Answer`로 최종 답 제출
- **채점**:
  - SELECT류(`other/counting/comparison/ranking/aggregation-*`): 정답 텍스트(집합) 비교
  - INSERT/UPDATE: 정답 SQL 실행 후 테이블 상태의 MD5 해시와, 에이전트 SQL 실행 후 테이블 상태의 해시를 비교
  - **Success Rate = mean(SELECT_accuracy, INSERT_accuracy, UPDATE_accuracy)**

## 실험 (원 논문, Table 3)
- 29개 LLM (API 기반 10개 + OSS 19개), DB 환경 300문제(standard split)
- `gpt-4` 32.0%, `gpt-3.5-turbo` 36.7%, `claude-2` 27.3% 등 — API 기반 모델이 OSS 모델(대부분 10~15%대)을 크게 앞섬
- 실패 유형 분석(Table 4): DB 태스크는 특히 **Invalid Format(포맷 미준수) 비율이 53.3%**로 8개 환경 중 가장 높음 — SQL 마크다운 형식을 못 맞추는 경우가 많다는 의미

## 우리 실험 (재현, Database 환경만 — 모델 3개 비교)

논문의 "상용 API가 오픈소스를 압도한다"는 2023년 결론이 지금도 유효한지 보기 위해,
서로 다른 코너(최저비용/최고신뢰/저비용오픈소스)를 대표하는 3개 모델을 비교했다.

| 모델 | 대표하는 코너 | SELECT | INSERT | UPDATE | Success Rate | 완료율 |
|---|---|---|---|---|---|---|
| `gpt-3.5-turbo` | 최저비용 기준선 | 26% | 37% | 71% | **44.67%** | 93.7% |
| `gpt-4o` | 최고신뢰 후보 | 32% | 32% | 78% | **47.33%** | 53.3%* |
| `llama-3.1-8b` (Groq, 오픈소스·무료) | 저비용 대량처리 후보 | 0% | 0% | 0% | **0%** | 0%† |

\* gpt-4o는 정확도 자체는 3개 중 최고지만, "agent context limit" 오분류(25.7%)로 완료율이 낮게 잡힘 — 아래 한계 참고
† llama-3.1-8b는 82.3%가 포맷 미준수(agent validation failed)로 즉시 실패 — 아래 한계 참고

- **데이터**: `dbbench-std`, 300문제 (논문과 동일 split), 3개 모델 각각 전체 실행
- 상세: `results/metrics.csv`, `results/raw/`

## 한계
- **재현 대상 축소**: 8개 환경 중 DB 1개만 재현 (OS 등 나머지는 커스텀 Docker 이미지 빌드 등 추가 작업 필요해 스코프 밖)
- **모델 버전 불일치**: 논문의 `gpt-3.5-turbo-0613`, `vicuna-13b`, `llama-2-13b` 등은 대부분 폐기되었거나 구식이라, 각 모델 계열의 **현재 시점 대표 모델**로 대체 (`gpt-3.5-turbo`, `gpt-4o`, `llama-3.1-8b`)
- **인프라 드리프트 이슈 (발견 1)**: `mysql:latest` 태그가 논문 작성 시점(2023, MySQL 8.0)과 지금(2026, MySQL 9.x) 사이에 이동하면서, INSERT/UPDATE 채점에 쓰이는 `MD5()` 함수 처리 방식이 달라져 **채점 자체가 실패하는 버그**를 확인. `mysql:8.0`으로 고정해 해결 (gpt-3.5-turbo INSERT 0%→37%, UPDATE 0%→71%)
- **에러 분류 로직의 한계 (발견 2)**: `gpt-4o`는 완료율이 53.3%까지 떨어지고 25.7%가 "agent context limit"으로 분류된다. 그러나 `gpt-4o`는 128k 토큰급 모델이라 실제 컨텍스트 초과일 가능성은 낮다. 프레임워크 코드(`http_agent.py`)의 `check_context_limit()`은 HTTP 에러 응답 텍스트에 `prompt/context/tokens` + `limit/exceed/max/...` 키워드가 동시에 있으면 무조건 컨텍스트 초과로 분류하는데, 이는 레이트리밋 등 다른 종류의 에러까지 오분류할 수 있는 허술한 휴리스틱이다. 이 예외는 원문을 로그에 남기지 않고 즉시 재발생(re-raise)하도록 짜여 있어, 정확한 원인은 특정하지 못했다 (추가 재현에는 재실행 비용 필요)
- **소형 오픈소스 모델의 포맷 미준수 (발견 3)**: `llama-3.1-8b`는 완료율 0%, 82.3%가 포맷 미준수로 실패. 실제 응답을 보면 지시된 고정 문구 `Action: Operation` 대신 `Action: Update`처럼 SQL 종류에 맞춰 말을 바꿔 쓰기 때문. 논문이 강조한 "약한 모델일수록 지시사항을 곧이곧대로 못 따른다"는 핵심 주장을 2026년 시점의 소형 오픈소스 모델로도 그대로 재현
- **평가 비용**: 논문처럼 29개 모델 전체 비교는 진행하지 않음 (API 비용/시간 제약으로 3개 모델로 스코프 축소)

## 코드 계획
- **실행**: `src/baseline.py` — 프레임워크 없이 LLM 1회 호출로 SQL 제안 확인, 3개 모델(`gpt-3.5-turbo`/`gpt-4o`/`llama-3.1-8b-instant`) 모두 지원
- **비교**: 원 논문 수치(gpt-3.5-turbo 36.7%) vs 우리 재현 수치(44.67%), 그리고 3개 모델 간 비교
- **분석**: `src/evaluate.py`로 원본 채점 로직을 독립적으로 재구현해, 프레임워크의 `overall.json`과 수치 일치 검증. 상태별(완료/라운드초과/포맷실패/컨텍스트초과) 분포도 함께 집계

## 프로젝트 활용
`docs/project_canvas.md` 참고 (팀 프로젝트 담당자가 별도로 작성 예정)
