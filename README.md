# PKB (Private Knowledge Base)

멀티-LLM 대화 내역을 정리된 지식 번들로 변환하는 CLI 도구.

[llm-chat-exporter](https://github.com/) Chrome 확장으로 내보낸 JSONL 파일을 입력받아, 메타데이터를 자동 생성하고 Obsidian에서 검색·열람 가능한 Markdown 번들을 만듭니다.

## 왜 PKB인가?

ChatGPT, Claude, Gemini, Grok, Perplexity — 여러 LLM에 같은 질문을 던지고 비교하는 워크플로우에서 **대화 후처리가 병목**입니다. 쌓여가는 대화 내역은 검색도 연결도 안 되는 "죽은 도서관"이 됩니다.

PKB는 이 문제를 해결합니다:

- **정리 시간 ≈ 0** — exporter 실행 후 `pkb ingest` 한 줄이면 메타데이터 자동 생성
- **검색 가능한 지식** — 태그 필터 + 시맨틱 검색으로 과거 대화를 되찾기
- **Obsidian 연동** — Markdown + frontmatter로 Dataview 쿼리 즉시 사용
- **자동 동기화** — Obsidian에서 메타 수정 → DB 자동 반영, 새 JSONL 감지 → 자동 인제스트

## 구조

PKB는 **도구**(이 리포)와 **데이터**(별도 KB 리포)를 분리합니다.

```
pkb/                            ← 도구 (이 리포지토리)
├── src/pkb/                    ← Python 패키지
├── prompts/                    ← LLM 프롬프트 템플릿
├── tests/                      ← 1342개 테스트 (1286 mock + 56 integration)
└── pyproject.toml

~/.pkb/                         ← 글로벌 설정 (자동 생성)
├── config.yaml                 ← KB 등록 + DB 설정 + LLM 설정
└── vocab/                      ← 공유 어휘 (domains, topics)

[원격 서버]                     ← DB 인프라
├── PostgreSQL                  ← 메타데이터 + FTS (tsvector/GIN)
└── ChromaDB                    ← 벡터 임베딩 (서버측 처리)

~/kb-personal/                  ← KB 리포 (데이터, 별도 git repo)
├── inbox/                      ← 감시 디렉토리 (pkb watch, 서브디렉토리 재귀 지원)
│   ├── PKB/                    ← exporter 디렉토리 복사 시 자동 감지
│   └── .done/                  ← 인제스트 완료 파일 (구조 보존)
└── bundles/{bundle_id}/
    ├── _raw/*.jsonl             ← 원본 JSONL (불변)
    ├── {platform}.md            ← 생성된 Markdown (derived)
    └── _bundle.md               ← 번들 메타데이터 (derived)
```

원본 JSONL은 불변(immutable)이고, 모든 Markdown과 메타데이터는 derived — 언제든 재생성 가능합니다.

## 설치

```bash
# Python 3.11+ 필요
git clone https://github.com/<your-username>/pkb.git
cd pkb

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

## 사용법

```bash
# 초기 설정 (~/.pkb/ 디렉토리 생성)
pkb init

# JSONL 파일 파싱 (구조 확인)
pkb parse conversation.jsonl

# 디렉토리 내 모든 JSONL 파싱
pkb parse ./exported/

# JSONL을 KB에 인제스트 (메타 자동 생성 + DB 저장)
pkb ingest conversation.jsonl --kb personal

# 디렉토리 일괄 인제스트
pkb ingest ./exported/ --kb personal

# 드라이런 (파일/DB 쓰기 없이 미리보기)
pkb ingest ./exported/ --kb personal --dry-run

# 대량 일괄 처리 (체크포인트 기반 재시작)
pkb batch ./all-exports/ --kb personal
pkb batch ./all-exports/ --kb personal --max 50  # 50개만

# 토픽 어휘 관리
pkb topics                    # 전체 보기 (하위 호환)
pkb topics list --status pending   # 미승인 토픽만
pkb topics approve <name>     # 토픽 승인
pkb topics merge <name> --into <target>  # 토픽 병합
pkb topics reject <name>      # 토픽 거부

# 검색 (하이브리드: FTS + 시맨틱)
pkb search "bitcoin halving"                      # 기본 하이브리드 검색
pkb search "bitcoin" --domain investing            # 도메인 필터
pkb search "async" --mode keyword                  # FTS 전용
pkb search "embedding" --mode semantic             # 시맨틱 전용
pkb search "python" --topic python --limit 5       # 토픽 필터 + 결과 제한
pkb search "test" --after 2026-01-01 --json        # 날짜 필터 + JSON 출력

# Obsidian에서 _bundle.md 수정 후 DB 동기화
pkb reindex 20260221-bitcoin-abc1 --kb personal    # 단일 번들
pkb reindex --full --kb personal                   # 전체 + 고아 정리

# 프롬프트/모델 변경 후 메타데이터 재생성
pkb regenerate 20260221-bitcoin-abc1 --kb personal  # 단일 번들
pkb regenerate --all --kb personal                  # 전체
pkb regenerate --all --kb personal --dry-run        # 미리보기

# 디렉토리 감시 → 새 파일 자동 인제스트 (서브디렉토리 + 시작 시 기존 파일 포함)
pkb watch                  # 모든 KB inbox 감시
pkb watch --kb personal    # 특정 KB만 감시

# 중복 번들 감지 (임베딩 유사도 기반)
pkb dedup scan --kb personal    # 중복 스캔
pkb dedup list                  # 중복 쌍 목록
pkb dedup dismiss <pair_id>     # 비중복 마킹
pkb dedup confirm <pair_id>     # 중복 확인

# 지식 그래프 (번들 관계)
pkb relate scan --kb personal   # 관계 스캔 (유사/관련/연속)
pkb relate list                 # 전체 관계 목록
pkb relate show <bundle_id>     # 번들별 관계 조회

# 지식 다이제스트 (주제/도메인 요약)
pkb digest --topic python       # 토픽 기반 지식 종합 요약
pkb digest --domain dev --kb n  # 도메인 기반 요약
pkb digest --topic ai -o out.md # 파일로 저장

# 로컬 웹 UI (FastAPI + htmx)
pkb web --port 8080             # 웹 서버 시작

# RAG 챗봇 (지식베이스 기반 대화)
pkb chat --kb personal          # 인터랙티브 챗봇 REPL
pkb chat --mode analyst         # 분석가 모드 (explorer/analyst/writer)

# MCP 서버 (Claude Code 연동)
pkb mcp-serve                   # stdio MCP 서버 시작

# KB 관리
pkb kb list                     # 등록된 KB 목록 + 번들 수

# 시스템 진단
pkb doctor                      # DB 연결, ChromaDB, LLM API 등 상태 점검
pkb doctor --skip-llm           # LLM API 체크 건너뛰기
pkb doctor --skip-db            # DB 체크 건너뛰기

# 데이터베이스 관리
pkb db upgrade                  # 최신 스키마로 업그레이드
pkb db downgrade 0001           # 특정 리비전으로 다운그레이드
pkb db current                  # 현재 리비전 확인
pkb db history                  # 마이그레이션 이력 확인
pkb db stamp head               # SQL 실행 없이 리비전 마킹
pkb db migrate-domain coding dev  # 도메인 이름 변경 (예: coding→dev)
pkb db migrate-stable-id --kb personal  # stable_id 재계산 (원본 파일 기반)
pkb db reset --kb personal      # KB 데이터 전체 삭제 (확인 필요)
```

## JSONL 입력 형식

[llm-chat-exporter](https://github.com/)가 출력하는 JSONL 형식을 사용합니다:

```jsonl
{"_meta":true,"platform":"claude","url":"...","exported_at":"2026-02-21T06:02:42.230Z","title":"..."}
{"role":"user","content":"...","timestamp":"..."}
{"role":"assistant","content":"...","timestamp":"..."}
```

첫 줄은 `_meta` 객체, 이후는 대화 턴입니다. Claude/Perplexity의 연속 assistant 턴(thinking + response)도 올바르게 처리합니다.

## 기술 스택

| 구성 요소 | 선택 | 이유 |
|-----------|------|------|
| LLM 라우팅 | 멀티 프로바이더 (Anthropic, OpenAI, Google, Grok) | Tier 기반 라우팅 + 에스컬레이션 |
| 구조화 DB | PostgreSQL (원격) | 메타 저장 + tsvector FTS, 네트워크 접근 |
| 벡터 DB | ChromaDB (원격) | 서버측 임베딩, HTTP API |
| 뷰어 | Obsidian + Dataview / 웹 UI | Markdown + frontmatter + FastAPI + htmx |
| 태그 체계 | 2-tier (Domain + Topic) | L1 고정 도메인 8개 + L2 controlled vocab |
| DB 마이그레이션 | Alembic | Raw SQL 기반 버전 관리, 롤백 지원 |
| 파일 감시 | watchdog | 5초 디바운스, JSONL + MD 필터 |
| 웹 UI | FastAPI + htmx + Jinja2 | 번들 관리, 검색, 토픽, 중복, 챗봇 |
| RAG 챗봇 | ChatEngine + LLMRouter | 지식베이스 기반 대화형 질의응답 |

## 개발

### Git Hooks 설정

최초 clone 후 반드시 실행:

```bash
git config core.hooksPath scripts/hooks
```

- `post-commit` — 커밋 시 자동 패치 버전 범프 + 빌드 (`release.sh patch`)
- `PKB_SKIP_POST_COMMIT=1`로 비활성화 가능

### 테스트

```bash
# 테스트
pytest                       # 1342개 테스트 (1286 mock + 56 integration)

# 통합 테스트 (Docker 필요)
docker compose -f docker/docker-compose.test.yml up -d
PKB_DB_INTEGRATION=1 pytest tests/integration/db/ -v

# 린트
ruff check src/ tests/       # line-length=100

# CLI 버전 확인
pkb --version
```

## 로드맵

| Phase | 상태 | 내용 |
|-------|------|------|
| **0** | ✅ 완료 | 프로젝트 스캐폴딩, JSONL 파서, 어휘 시드 |
| **1** | ✅ 완료 | 인제스트 파이프라인 (`pkb ingest` → MD + frontmatter + _bundle.md + PostgreSQL + ChromaDB) |
| **2** | ✅ 완료 | 검색 레이어 (`pkb search` — 하이브리드 FTS + 시맨틱, 가중 점수 0.4:0.6) |
| **3** | ✅ 완료 | 자동화 (`pkb reindex` frontmatter 동기화, `pkb regenerate` LLM 재추출, `pkb watch` 자동 인제스트) |
| **4** | ✅ 완료 | 토픽 CLI, 중복 감지, LLM 라우팅, 웹 UI (FastAPI + htmx), RAG 챗봇 |
| **5** | ✅ 완료 | 지식 그래프 (`bundle_relations`, `pkb relate` CLI, 관계 웹 UI, D3.js 그래프 API) |
| **6** | ✅ 완료 | 스마트 어시스턴트 (`DigestEngine`, 대화 모드 explorer/analyst/writer, MCP 서버 `pkb mcp-serve`, 다이제스트 웹 UI) |
| **7** | ✅ 완료 | 분석 대시보드 (`AnalyticsEngine`, `ReportGenerator`, `pkb stats`/`pkb report` CLI, Chart.js 웹 대시보드) |
| **8** | ✅ 완료 | 자동화 파이프라인 (`PostIngestProcessor` 자동 관계/중복/갭 분석, 메타데이터 활용 강화, `Scheduler` 주기적 작업) |
| **9** | ✅ 완료 | MCP 확장 (4→14 tools: ingest, browse, detail, graph, gaps, claims, timeline, recent, compare, suggest) |
| **10** | ✅ 완료 | Web UI 강화 (D3.js 지식 그래프, 인사이트 대시보드, LLM 비교 뷰, Chat 3-panel) |

자세한 설계는 [`docs/design-v1.md`](docs/design-v1.md)를 참조하세요.

## 라이선스

Private project.
