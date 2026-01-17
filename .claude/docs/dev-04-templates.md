# GitHub 템플릿 가이드

> PR 및 Issue 템플릿 사용법

## 개요

프로젝트에서는 일관된 PR과 Issue 작성을 위해 템플릿을 제공합니다.

### 파일 구조
```
.github/
├── PULL_REQUEST_TEMPLATE.md      # PR 템플릿
└── ISSUE_TEMPLATE/
    ├── bug_report.yml            # 버그 리포트 (YAML Form)
    ├── feature_request.yml       # 기능 요청 (YAML Form)
    └── config.yml                # 템플릿 설정
```

## Pull Request 템플릿

### 사용 방법
PR 생성 시 자동으로 템플릿이 로드됩니다.

### 섹션 설명

| 섹션 | 필수 | 설명 |
|------|------|------|
| Summary | ✅ | 변경 사항 1-3문장 요약 |
| Related Issue | ✅ | `Closes #123` 형식으로 이슈 연결 |
| Type of Change | ✅ | 변경 유형 선택 |
| Changes Made | ✅ | 주요 변경 사항 리스트 |
| Screenshots | ❌ | UI 변경 시 스크린샷 |
| Test Plan | ✅ | 테스트 방법 설명 |
| Checklist | ✅ | 제출 전 확인 사항 |
| Breaking Changes | ❌ | Breaking change 설명 |

### 좋은 PR 작성법

1. **제목**: Conventional Commits 형식
   ```
   feat(auth): add OAuth2 login
   fix(api): resolve null pointer exception
   ```

2. **작은 단위**: 50-200줄, 1-5개 파일

3. **명확한 설명**: Why > What

4. **Self-review**: 제출 전 직접 리뷰

## Issue 템플릿

### 버그 리포트 (`bug_report.yml`)

YAML 기반 폼으로 구조화된 정보 수집:

- **버그 설명**: 문제 상황 명확히
- **재현 방법**: 단계별 재현 경로
- **예상 동작**: 정상 동작 설명
- **실제 동작**: 발생한 문제
- **컴포넌트**: 관련 모듈 선택
- **환경 정보**: OS, 버전 등
- **로그/에러**: 관련 로그 첨부

### 기능 요청 (`feature_request.yml`)

- **문제**: 해결하려는 문제
- **해결책**: 제안하는 기능
- **대안**: 고려한 다른 방법
- **컴포넌트**: 관련 모듈
- **우선순위**: 중요도

## 라벨 시스템

### 자동 할당 라벨

| 템플릿 | 라벨 |
|--------|------|
| 버그 리포트 | `bug`, `triage` |
| 기능 요청 | `enhancement`, `triage` |

### 추가 라벨 권장

| 라벨 | 용도 |
|------|------|
| `priority:high` | 긴급 |
| `priority:medium` | 중간 |
| `priority:low` | 낮음 |
| `component:api` | API 서비스 |
| `component:ml` | ML 서비스 |
| `good first issue` | 입문자용 |
| `help wanted` | 도움 필요 |

## Best Practices

### PR 작성 시
- 하나의 PR = 하나의 논리적 변경
- 리뷰어를 고려한 설명
- 테스트 포함

### Issue 작성 시
- 중복 이슈 먼저 검색
- 가능한 자세히 작성
- 재현 가능한 정보 제공

### 리뷰 시
```
🔴 [MUST] 필수 수정
🟡 [SHOULD] 권장
🟢 [COULD] 제안
❓ [Q] 질문
👍 [NICE] 칭찬
```

## 참고 자료

- [GitHub PR Template Guide](https://axolo.co/blog/p/part-3-github-pull-request-template)
- [GitHub Issue Forms](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository)
- [Graphite PR Checklist](https://graphite.com/guides/comprehensive-checklist-github-pr-template)

---

*관련 문서: `git-01-rules.md`, `09-CI_CD.md`*
