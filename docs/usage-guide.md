# PKB 사용 가이드

## 1. 초기 셋업 (한 번만)

```bash
pkb init                    # ~/.pkb/ 디렉토리 + config.yaml + vocab 생성
pkb doctor                  # DB 연결, ChromaDB, LLM API 상태 점검
pkb db upgrade              # DB 스키마 최신으로 마이그레이션
```

`pkb doctor`는 PostgreSQL, ChromaDB, LLM API 키를 한 번에 검증한다. 셋업 후 반드시 한번 돌려서 연결 상태를 확인할 것. `--skip-llm`, `--skip-db`로 부분 점검도 가능.

## 2. 일상 워크플로우: 대화 → 지식 번들

### 2-1. 단건 인제스트

```bash
# 1) 먼저 파싱으로 내용 확인
pkb parse conversation.jsonl

# 2) 인제스트 (dry-run으로 미리보기 가능)
pkb ingest conversation.jsonl --kb personal --dry-run
pkb ingest conversation.jsonl --kb personal
```

### 2-2. 디렉토리 일괄 인제스트

```bash
pkb ingest ./exported/ --kb personal
```

### 2-3. 대량 처리 (batch)

```bash
# 체크포인트 기반 — 중단 후 재시작 가능
pkb batch ./all-exports/ --kb personal
pkb batch ./all-exports/ --kb personal --max 50      # 50개만
pkb batch ./all-exports/ --kb personal --workers 4   # 병렬 4워커
```

- `ingest` vs `batch`: 단건/소량은 `ingest`, 대량은 `batch`. `batch`는 체크포인트를 저장해서 중단 후 이어서 처리 가능.
- 성공한 파일은 자동으로 `inbox/.done/`으로 이동하므로 재처리 걱정 없음.
- 같은 대화의 다른 플랫폼 파일(claude.jsonl + chatgpt.jsonl)은 자동 머지됨.

### 2-4. 자동 인제스트 (watch 모드)

```bash
pkb watch                   # 모든 KB inbox 감시
pkb watch --kb personal     # 특정 KB만
```

서브디렉토리도 재귀 감시하므로, exporter 폴더를 그대로 inbox에 복사하면 된다. 기존 파일도 시작 시 자동 스캔.

## 3. 검색

```bash
# 기본 하이브리드 검색 (FTS 0.4 + 시맨틱 0.6)
pkb search "bitcoin halving"

# 필터 조합
pkb search "async" --mode keyword              # FTS 전용
pkb search "embedding" --mode semantic          # 시맨틱 전용
pkb search "python" --domain dev --topic python # 도메인 + 토픽 필터
pkb search "test" --after 2026-01-01 --limit 5  # 날짜 + 개수 제한

# 스크립트/자동화용
pkb search "query" --json
```

- `hybrid`가 기본이고 대부분의 경우 가장 좋은 결과를 줌. 정확한 키워드가 있으면 `keyword`, 의미 기반 탐색이면 `semantic`.
- `--domain`과 `--topic`은 반복 가능 (여러 개 AND 필터).

## 4. Obsidian 연동

번들 디렉토리(`~/kb-personal/bundles/`)를 Obsidian Vault로 열면:
- 각 `_bundle.md`의 YAML frontmatter를 Dataview 쿼리로 활용 가능
- Obsidian에서 frontmatter를 수정한 후 `pkb reindex`로 DB에 반영:

```bash
pkb reindex 20260221-bitcoin-abc1 --kb personal   # 단일
pkb reindex --full --kb personal                   # 전체 + 고아 레코드 정리
```

## 5. 메타데이터 관리

### 5-1. 토픽 어휘 관리

LLM이 자동 생성한 토픽은 `pending` 상태로 시작한다:

```bash
pkb topics                         # 전체 보기
pkb topics list --status pending   # 미승인만
pkb topics approve python          # 승인
pkb topics merge py --into python  # 병합 (별칭 통합)
pkb topics reject typo-topic       # 거부/삭제
```

### 5-2. 메타데이터 재생성

프롬프트나 모델을 변경한 후:

```bash
pkb regenerate 20260221-bitcoin-abc1 --kb personal --dry-run  # 미리보기
pkb regenerate --all --kb personal                            # 전체 재생성
```

## 6. 중복 및 관계 관리

### 6-1. 중복 감지

```bash
pkb dedup scan --kb personal       # 임베딩 유사도 기반 스캔
pkb dedup list                     # 중복 후보 목록
pkb dedup dismiss 42               # 비중복 처리
pkb dedup confirm 42               # 중복 확인
```

### 6-2. 지식 그래프

```bash
pkb relate scan --kb personal      # 번들 간 관계 발견 (similar/related)
pkb relate list                    # 전체 관계
pkb relate show 20260221-bitcoin-abc1  # 특정 번들의 관계
```

## 7. 지식 활용

### 7-1. 다이제스트 (종합 요약)

```bash
pkb digest --topic python              # 토픽 기반 지식 종합
pkb digest --domain dev --kb personal  # 도메인 기반
pkb digest --topic ai -o ai-digest.md  # 파일로 저장
```

### 7-2. RAG 챗봇

```bash
pkb chat --kb personal              # 기본 explorer 모드
pkb chat --mode analyst             # 분석가 모드 (심층 분석)
pkb chat --mode writer              # 작성자 모드 (콘텐츠 생성)
```

3가지 챗 모드:
- **explorer**: 탐색형 — 넓게 관련 번들을 찾아 답변
- **analyst**: 분석형 — 데이터 기반 심층 분석
- **writer**: 작성형 — 지식을 바탕으로 콘텐츠 초안 작성

### 7-3. MCP 서버 (Claude Code 연동)

```bash
pkb mcp-serve    # Claude Code에서 PKB 지식베이스를 직접 검색/질의
```

### 7-4. 웹 UI

```bash
pkb web --port 8080    # 브라우저에서 번들 관리, 검색, 토픽, 챗봇, 대시보드
```

## 8. 분석 및 리포트

```bash
pkb stats                          # KB 개요 (번들 수, 도메인, 토픽)
pkb stats --domain                 # 도메인별 분포
pkb stats --json                   # JSON 출력
pkb report                         # 주간 활동 리포트
pkb report --period monthly        # 월간 리포트 (지식 갭 분석 포함)
pkb report -o report.md            # 파일 저장
```

## 9. 추천 일상 루틴

| 타이밍 | 할 일 |
|--------|-------|
| **매일** | exporter로 내보내기 → inbox에 복사 (watch 모드 켜두면 자동 처리) |
| **주 1회** | `pkb topics list --status pending` → approve/merge/reject |
| **주 1회** | `pkb dedup scan` + `pkb relate scan` → 중복 정리 + 관계 갱신 |
| **주 1회** | `pkb report` → 지난 주 활동 리뷰 |
| **필요 시** | `pkb digest --topic X` → 특정 주제 지식 종합 |
| **필요 시** | `pkb chat` → KB 기반 질의응답 |

## 10. 트러블슈팅

```bash
pkb doctor                  # 전체 시스템 진단
pkb doctor --skip-llm       # LLM 없이 DB만 체크
pkb -v ingest ...           # INFO 로깅
pkb -vv ingest ...          # DEBUG 로깅 (상세)
pkb db current              # 현재 DB 마이그레이션 상태
pkb db history              # 마이그레이션 이력
```

**핵심 원칙**: 원본 JSONL은 불변. 모든 MD/frontmatter/_bundle.md는 derived(파생물)이므로 `pkb regenerate`로 언제든 재생성 가능. 뭔가 꼬였다면 `regenerate --all`이 최후의 수단.

## 요약

가장 효율적인 사용법:
1. `pkb watch`를 상시 켜두기
2. exporter 결과물을 inbox에 복사하면 자동 인제스트
3. 주기적으로 토픽 정리 + 중복/관계 스캔
4. 필요할 때 `search`, `chat`, `digest`로 지식 활용
