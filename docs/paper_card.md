# Paper Card

> 패기 3팀 | 박봄이(팀장), 김서연, 박수빈, 백종윤, 원준서, 황지현

| 항목 | 작성 내용 |
|---|---|
| **논문 정보** | **AgentBench: Evaluating LLMs as Agents**<br>Xiao Liu, Hao Yu, Hanchen Zhang et al. \| ICLR 2024 \| arXiv:2308.03688<br>Code · Data: [https://github.com/THUDM/AgentBench](https://github.com/THUDM/AgentBench) |
| **한 줄 요약** | 상호작용 환경에서 LLM이 목표를 달성하는 능력을 종합적으로 평가하기 위한 다차원 벤치마크 **AgentBench**를 제안한다. |
| **문제정의** | 기존 LLM 평가는 단발성 응답 정확도 중심으로, 행동 선택·환경 피드백 반영·멀티턴 목표 수행 능력을 충분히 평가하지 못한다. 기존 Agent 벤치마크도 폐쇄적 행동 공간 또는 단일 환경에 한정된다. |
| **핵심 아이디어** | OS·DB·KG·게임·웹 등 8개 상호작용 환경에서 **Thought → Action → Observation** 루프를 반복하며 수행 성능을 평가하고, 환경별 점수를 정규화하여 **Overall AgentBench Score**로 종합 비교한다. |
| **모델/알고리즘** | **입력** :  사용자 지시, 이전 상호작용 기록, 환경 결과<br>**출력** : 추론과 하나의 행동 또는 최종 답변<br>**구조** : Agent Server–Evaluation Client–Task Server<br>**실행 과정** : Action 실행 → Observation 반환 → 기록 갱신 → 성공 또는 턴 제한까지 반복<br>**종합 점수** : $OA(M)=\frac{1}{8}\sum_{i=1}^{8}w_iS_{M,i}$<br>($w_i$: 환경 가중치, $S_{M,i}$: 모델 $M$의 환경 $i$ 점수)<br>**Edmonds–Karp 알고리즘** : Agent–Task Worker 매칭(작업 배분)에 사용 |
| **실험** | **데이터** : 8개 환경(코드·게임·웹 기반), 테스트 1,014개 작업<br>**비교 모델** : 상용 API 10개 + 오픈소스 19개(총 29개 LLM)<br>**지표:** SR, F1, Reward, Win Rate, Game Progress 등 및 Overall AgentBench Score<br>**결과** : GPT-4가 OA 4.01로 최고, 상용 평균 2.32 vs 오픈소스 평균 0.51<br>모델 크기와 성능은 비례하지 않으며, TLE가 주요 실패 원인<br>코드 학습은 정적 절차 과제에는 도움을 주지만, 복잡한 판단 과제에서는 성능이 하락<br>대화 기반 정렬 데이터 학습이 성능 향상에 효과적 |
| **한계** | 2023년 시점의 단일 Agent 중심 작업으로, 장기 계획·Multi-Agent·Memory·Security 평가에 한계가 있다.<br>입력 기록을 3,500토큰으로 절단하며, 환경별 지표와 가중치가 종합 점수에 영향을 준다.<br>일부 생성 데이터 및 자동 평가의 편향 가능성, API 비용 및 재현성 문제가 있다. |
| **코드 계획** | AgentBench-Mini-Eval의 DB(MySQL) · LTP 환경 재현<br>실제 실행 결과를 반복 제공하며 GPT·Claude·Llama·Vicuna 계열 비교<br>**DB** : SR 및 명령 유형별 성능<br>**LTP** : GP·SGA, 완료율 · TLE · 포맷 오류 분석 |
| **프로젝트 활용** | **헌혈 예약 AI Agent 평가 환경 구축**<br>가상 예약 DB와 위치 조회 · 헌혈 주기 검증·가용 슬롯 확인 · 예약 확정 Tool 제공<br>정보 누락 시 역질문, 최대 8턴 제한, 실패 복구 로직 적용<br>**평가 지표** : 예약 성공률, 평균 API 호출 턴 수, TLE, 정책 위반율(Policy Violation Rate) |

