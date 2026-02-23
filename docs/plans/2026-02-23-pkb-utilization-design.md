# PKB 지식 활용 시스템 설계 — Phase 5/6/7

> 1,953+ 번들이 축적된 PKB를 "죽은 도서관"에서 "살아있는 지식 시스템"으로 전환하기 위한 설계.

---

## 1. 문제 정의

### 현재 상태
- Phase 0~4 완료: ingest, search, watch, dedup, web UI, RAG chat 모두 구축
- 1,953개 번들 (8개 도메인, 다수 토픽)
- 사용 중인 인터페이스: Obsidian, CLI search, RAG chat, Web UI

### Pain Points
- **지식 연결/통합 부족**: 번들이 개별 문서로만 존재. 관련 번들 간 연결이 없음
- **활용 방법 자체가 모호**: 데이터는 쌓이지만 이를 "가치"로 전환하는 방법이 부족

### 목표 가치
1. **의사결정 지원**: 특정 주제에 대해 과거에 모은 정보를 종합하여 판단에 활용
2. **지식 재활용/학습**: 과거 학습 내용을 맥락과 함께 다시 떠올려 깊은 이해로 발전
3. **콘텐츠 생산**: 축적된 지식을 기반으로 글, 보고서, 노트 등 새로운 콘텐츠 생성
4. **트렌드/패턴 발견**: 관심사의 변화, 반복 주제, 지식 공백 등을 시각적으로 파악

---

## 2. 전체 아키텍처

```
Phase 5: Knowledge Graph (기반)
    ↓ 관계 데이터
Phase 6: Smart Assistant (활용)
    ↓ 분석 결과
Phase 7: Analytics Dashboard (분석)
```

### 인터페이스 지원
- CLI: 모든 기능의 primary interface
- Web UI: 시각화 + 대시보드 (FastAPI + htmx 확장)
- Obsidian: 그래프 뷰 + wikilink 연동
- Claude Code: MCP 서버 통합

---

## 3. Phase 5: Knowledge Graph — 지식 연결망 구축

### 3.1 Bundle Relations (번들 간 관계)

**관계 유형**:
| Type | 기준 | 설명 |
|------|------|------|
| `similar` | 임베딩 유사도 ≥ threshold | 내용적으로 유사한 번들 |
| `related` | 토픽/도메인 공유 | 같은 주제를 다룬 번들 |
| `sequel` | 같은 토픽 + 시간순 | 동일 주제의 후속 대화 |

**DB 스키마** (`bundle_relations` 테이블):
```sql
CREATE TABLE bundle_relations (
    id SERIAL PRIMARY KEY,
    source_bundle_id VARCHAR(128) NOT NULL REFERENCES bundles(bundle_id),
    target_bundle_id VARCHAR(128) NOT NULL REFERENCES bundles(bundle_id),
    relation_type VARCHAR(20) NOT NULL,  -- similar, related, sequel
    score FLOAT NOT NULL,                -- 관계 강도 (0.0 ~ 1.0)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_bundle_id, target_bundle_id, relation_type)
);
CREATE INDEX idx_relations_source ON bundle_relations(source_bundle_id);
CREATE INDEX idx_relations_target ON bundle_relations(target_bundle_id);
CREATE INDEX idx_relations_type ON bundle_relations(relation_type);
```

**관계 구축 프로세스**:
1. ChromaDB에서 각 번들의 chunk 임베딩 조회
2. 번들 간 평균 임베딩 유사도 계산 → `similar` 관계
3. PostgreSQL에서 공유 토픽/도메인 쿼리 → `related` 관계
4. 같은 토픽 + 날짜 순서 → `sequel` 관계

### 3.2 Knowledge Map (Web UI 시각화)

- **D3.js force-directed graph**: 노드(번들), 엣지(관계)
- 노드 색상 = 도메인, 크기 = 관계 수
- 필터: 도메인, 토픽, 시간 범위, 관계 유형
- 클릭 → 번들 상세 + 관련 번들 목록 사이드패널

### 3.3 Obsidian Integration

- `_bundle.md` frontmatter에 `related:` 필드 자동 추가
- `[[20260221-bitcoin-halving-a3f2]]` 형식의 wikilink
- Obsidian Graph View에서 자연스럽게 연결망 시각화

### 3.4 CLI

```bash
pkb relate <bundle_id>              # 관련 번들 탐색
pkb relate --build                  # 전체 관계 일괄 구축
pkb relate --build --type similar   # 특정 유형만 구축
pkb graph --topic <topic>           # 특정 토픽의 연결 맵
pkb graph --domain <domain>         # 도메인별 연결 맵
```

---

## 4. Phase 6: Smart Assistant — 지식 비서 고도화

### 4.1 Advanced RAG Queries

현재 `pkb chat`의 단순 Q&A를 넘어서:

| Query Type | 예시 | 동작 |
|------------|------|------|
| 종합 분석 | "블록체인에 대해 내가 아는 것 전체를 요약해줘" | 토픽 관련 모든 번들 수집 → 종합 |
| 시간 비교 | "6개월 전과 지금 AI에 대한 내 관점 변화" | 시간순 정렬 → 변화 분석 |
| 크로스 LLM | "이 주제에 대해 GPT와 Claude 비교" | 플랫폼별 필터 → 비교 분석 |
| 의사결정 | "테슬라 투자 pros/cons 정리" | 관련 번들 → pros/cons 추출 |

### 4.2 Smart Digest (주제별 종합 리포트)

```bash
pkb digest --topic "blockchain"     # 토픽별 종합 리포트
pkb digest --domain "투자"          # 도메인별 종합
pkb digest --topic "ai" --since 2026-01  # 기간 지정
```

- Phase 5의 관계 그래프를 활용해 관련 번들도 함께 수집
- LLM으로 종합 요약 생성
- 마크다운 리포트 출력 (파일 저장 or stdout)

### 4.3 Claude Code MCP Server

PKB를 MCP 서버로 노출:

```yaml
# .mcp.json
{
  "mcpServers": {
    "pkb": {
      "command": "pkb",
      "args": ["mcp-serve"],
      "env": {}
    }
  }
}
```

**MCP Tools**:
- `pkb_search` — 지식 검색 (query, mode, domain, topic)
- `pkb_related` — 관련 번들 조회 (bundle_id)
- `pkb_digest` — 주제별 종합 (topic/domain)
- `pkb_stats` — KB 통계 조회

### 4.4 Conversation Modes

`pkb chat`에 모드 선택:
- **Explorer** (기본): 자유 탐색, 현재와 동일
- **Analyst**: 분석/비교/트렌드 질의에 최적화된 시스템 프롬프트
- **Writer**: 콘텐츠 초안 생성, 아웃라인 작성 지원

```bash
pkb chat --mode analyst --kb personal
pkb chat --mode writer --topic "blockchain"
```

---

## 5. Phase 7: Analytics Dashboard — 지식 분석

### 5.1 Knowledge Portfolio

- **도메인 분포**: 8개 도메인별 번들 수 파이/도넛 차트
- **토픽 히트맵**: Top 20 토픽, 빈도 + 최근 활동 기준
- **시간축 트렌드**: 월별 도메인/토픽 변화 추이 (라인 차트)
- **LLM 사용 분포**: 플랫폼별 사용 빈도 (바 차트)

### 5.2 Knowledge Gap Detection

- 번들 수 1~2개인 토픽 식별 → "한 번만 물어본 주제"
- 관계 그래프에서 고립된 번들(edge 0) 탐지
- 추천 메시지: "이 토픽은 더 깊이 파보면 좋겠습니다"

### 5.3 Periodic Digest

```bash
pkb report --weekly                 # 이번 주 활동 요약
pkb report --monthly                # 월간 리포트
pkb report --monthly --since 2026-01  # 특정 월
```

리포트 내용:
- 새로 추가된 번들 수 + 목록
- 활발한 토픽 Top 5
- 도메인 분포 변화
- 새로 등장한 토픽

### 5.4 Web Dashboard

- 기존 FastAPI dashboard 페이지 확장
- Chart.js로 인터랙티브 차트
- 필터: 기간, 도메인, KB 선택
- `/analytics` 라우트 추가

### 5.5 CLI

```bash
pkb stats                           # 전체 통계 요약
pkb stats --domain dev              # 도메인별 상세
pkb stats --trend --months 6        # 6개월 트렌드
pkb report --weekly                 # 주간 리포트
```

---

## 6. 구현 순서 요약

| Phase | 이름 | 핵심 산출물 | 의존성 |
|-------|------|------------|--------|
| 5 | Knowledge Graph | `bundle_relations` 테이블, 관계 구축기, 그래프 시각화, Obsidian wikilink | Phase 1~4 |
| 6 | Smart Assistant | 고급 RAG, Digest, MCP 서버, Conversation Modes | Phase 5 (관계 데이터) |
| 7 | Analytics Dashboard | 포트폴리오 차트, Gap 탐지, 정기 리포트, Web 대시보드 | Phase 5 (관계), Phase 6 (digest) |

---

## 7. 기술 스택 추가

| 구성요소 | 기술 | 용도 |
|----------|------|------|
| 그래프 시각화 | D3.js (force-directed) | Knowledge Map |
| 차트 | Chart.js | Analytics Dashboard |
| MCP 서버 | Python MCP SDK (`mcp` package) | Claude Code 통합 |
| Obsidian 연동 | frontmatter wikilink | Graph View |
