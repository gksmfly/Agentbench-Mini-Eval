# AgentBench: Evaluating LLMs as Agents — Q&A 메모

패기3팀 논문 발표 대비 예상 질문·답변 · 실패 사례 · 한계 정리

---

## 1. 예상 질문과 답변

**Q. AgentBench의 핵심 기여는 무엇인가?**
A. OS · DB · KG · DCG · LTP · HH · WS · WB 8개 실제 환경을 아우르는 최초의 종합 LLM 에이전트 평가 벤치마크를 제안하고, 상용 10개·오픈소스 19개 총 29개 모델을 통일된 기준(OA score)으로 비교했다.

**Q. 기존 벤치마크(MMLU, HumanEval)와 무엇이 다른가?**
A. MMLU · HumanEval은 단발성(Single-turn) 응답의 정답률만 측정하지만, AgentBench는 Action → Observation → Next Action 루프를 반복하며 환경 결과를 반영해 목표를 달성하는 멀티턴 상호작용 능력을 평가한다.

**Q. 상용 모델이 오픈소스보다 훨씬 우수한 이유는?**
A. OA 평균이 상용 2.32 대 오픈소스 0.51로 약 4배 차이. 오픈소스는 Invalid Format / Invalid Action / TLE 비율이 높아 포맷 준수·지시 이행·장기 추론에서 취약하다.

**Q. 파라미터 크기가 성능과 비례하지 않는 이유는?**
A. 산점도 분석 결과 동일 계열에서도 크기 증가가 OA 향상으로 이어지지 않는다. Agent 능력은 지식량보다 코드 학습, 대화 기반 정렬(지시 이행·멀티턴 능력) 같은 별도 학습 요인에 좌우된다.

**Q. 실패 유형(Invalid Format / Action / TLE)이 환경마다 다른 이유는?**
A. DB·DCG는 JSON/SQL 등 엄격한 포맷을 요구해 Format 오류가 집중되고, HH는 자유로운 행동 공간이라 존재하지 않는 장소 탐색 같은 Invalid Action이 많으며, 그 외 환경은 유의미한 진전 없이 유사 행동을 반복해 TLE(라운드 소진)로 귀결된다.

**Q. 재구현 실험에서 Claude-Sonnet-5가 DB에서 가장 안정적인 이유는?**
A. SR 68%, 완료율 97%, SQL 실행 에러율 7.3%(gpt-4o와 동급, gpt-3.5-turbo 51.7%·vicuna 64.3% 대비 훨씬 낮음)로 안정적으로 수행. 반면 LTP에서는 완료율은 최고(20건 중 13건, 65%)지만 SGA(질문 유효성)는 17.2%로 낮아, 반복은 잘하지만 효율적인 질문 설계는 상대적으로 약하다.

**Q. Llama-3.1-8B가 DB·LTP에서 모두 완료율 0%인 이유는?**
A. 두 환경의 실패 원인은 서로 다르다. DB에서는 SQL 포맷 위반으로 82.3%가 agent validation failed 처리되어 Success Rate 0%. LTP에서는 SQL/JSON과 무관하게, 매 라운드 실제 질문을 던지지 못하고 "질문해주세요" 같은 빈 응답만 반복하다 25라운드를 소진해 100% TLE(task limit reached)로 종료됐다. 두 경우 모두 소형 오픈소스 모델의 지시 이행력 한계를 보여주지만, 원인은 "포맷 위반"(DB)과 "실질적 진행 실패로 인한 TLE"(LTP)로 다르다.

**Q. 헌혈 예약 AI 에이전트 프로젝트와 AgentBench의 연결점은?**
A. 실시간 DB 상태 반영, 정보 부족 시 역질문을 통한 Multi-turn 정보 보완, TLE·최대 턴 제한을 통한 무한 루프 방지 등 AgentBench의 Action-Observation 루프와 실패 회복 철학을 실제 서비스 설계에 적용했다.

**Q. AgentBench 이후 연구 흐름에서 이 논문의 위치는?**
A. 단일 Tool·짧은 계획·단일 Agent 중심이던 2023년 초기 표준을 확립했으나, 현재는 Long-Horizon 계획, Multi-Agent 협업, Reliability & Safety, Memory 등으로 다각화되어 AgentBench는 그 출발점으로 평가된다.

---

## 2. 실패 사례 (Failure Cases)

| 구분 | 환경/모델 | 실패 유형 | 수치 | 원인 |
|---|---|---|---|---|
| 원논문 | DB | Invalid Format | 53.3% (실패 비율) | CoT 형식·JSON 문법 위배 |
| 원논문 | DCG | Invalid Format | 38.5% (실패 비율) | CoT 형식·JSON 문법 위배 |
| 원논문 | HH | Invalid Action | 64.1% (실패 비율) | 존재하지 않는 장소 탐색 등 유효하지 않은 행동 |
| 원논문 | LTP | TLE | 82.5% (실패 비율) | 진전 없이 유사 행동 반복, 최대 라운드 초과 |
| 원논문 | KG | TLE | 67.9% (실패 비율) | 진전 없이 유사 행동 반복, 최대 라운드 초과 |
| 원논문 | WB | TLE | 35.0% (실패 비율) | 진전 없이 유사 행동 반복, 최대 라운드 초과 |
| 재구현 | Llama-3.1-8B (DB) | Invalid Format | 0% (완료율) | SQL 문법 위반으로 82.3%가 agent validation failed |
| 재구현 | Llama-3.1-8B (LTP) | TLE | 0% (완료율) | 실제 질문 없이 빈 응답 반복, 100% task limit reached |
| 재구현 | Vicuna-13B (DB) | 응답 품질 저하 | 22.7% (완료율) | 응답 장황·SQL 에러 반복 |
| 재구현 | GPT-4o (DB) | 컨텍스트 초과·포맷 위반 | 53.3% (완료율) | 컨텍스트 초과 25.7%, agent validation failed 21%로 두 원인이 비슷한 비중 |

---

## 3. 한계 (Limitations)

- 논문 발표 시점(2023) 기준 Agent로 설계되어 현재의 Long-Horizon 계획·Multi-Agent 협업·복합 Tool 체이닝 능력은 평가하기 어려움
- 평가 대상이 GPT-4-0613 등 2023년 모델 중심이라 최신 모델 성능은 별도 검증 필요
- 재구현 실험은 5개 모델 중심(DB 300건, LTP 20건)으로, LTP는 원논문 대비 표본이 적어 일반화에 한계
- prompt injection·tool abuse 등 보안·신뢰성 측면은 원논문 범위 밖
- 정적으로 구성된 태스크 세트라 실제 서비스의 동적 환경 변화를 완전히 반영하지는 못함
