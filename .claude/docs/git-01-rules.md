# Git 지침서 (1/2) - 규칙

## 1. 브랜치 전략 (택1)

### A) GitHub Flow (권장)
- main = 항상 배포 가능
- feature 브랜치 → PR → main 머지
- **적합**: 소규모팀, CI/CD, 빠른 배포

```
main ← feature/JIRA-123-login
     ← fix/JIRA-124-bug
     ← hotfix/JIRA-125-critical
```

### B) Git Flow
- main = 프로덕션, develop = 개발 통합
- feature → develop → release → main
- **적합**: 대규모팀, 정기 릴리스

```
main ← release ← develop ← feature/JIRA-123
                         ← hotfix/JIRA-124
```

### 선택 가이드
| 상황 | 추천 |
|------|------|
| 1-5명, CI/CD | GitHub Flow |
| 대기업, 정기 릴리스 | Git Flow |
| 스타트업/MVP | GitHub Flow |
| 다중 버전 유지보수 | Git Flow |

## 2. 브랜치 네이밍
```
{type}/{ticket}-{description}
```

| Type | 용도 | 예시 |
|------|------|------|
| `feature` | 새 기능 | `feature/JIRA-123-user-login` |
| `fix` | 버그 수정 | `fix/GH-456-auth-error` |
| `hotfix` | 긴급 수정 | `hotfix/ISSUE-789-security` |
| `refactor` | 리팩토링 | `refactor/DEV-101-cleanup` |
| `docs` | 문서 | `docs/JIRA-102-api-readme` |
| `chore` | 설정/빌드 | `chore/JIRA-103-ci` |

### 규칙
- 소문자 (티켓번호는 대문자 허용)
- 단어 구분: kebab-case
- 티켓 번호 필수
- 3-4 단어 이내

## 3. 커밋 메시지 (Conventional Commits)
```
<type>(<scope>): <subject>
```

### Type
| Type | 설명 | 예시 |
|------|------|------|
| `feat` | 새 기능 | `feat(auth): add OAuth2 login` |
| `fix` | 버그 수정 | `fix(api): resolve null pointer` |
| `docs` | 문서 변경 | `docs(readme): update guide` |
| `style` | 포맷팅 | `style(lint): fix warnings` |
| `refactor` | 리팩토링 | `refactor(auth): simplify logic` |
| `test` | 테스트 | `test(auth): add unit tests` |
| `chore` | 빌드/설정 | `chore(ci): add Actions` |
| `perf` | 성능 개선 | `perf(db): optimize query` |
| `ci` | CI 설정 | `ci(github): add deploy` |

### Subject 규칙
- 영어, 소문자 시작
- 명령형 (add, fix, update)
- 마침표 없음
- 50자 이내

### Body (선택)
```
feat(auth): add multi-factor authentication

- Add TOTP support for 2FA
- Integrate with Authy API

Closes #123
```

### Breaking Change
```
feat(api)!: change response format

BREAKING CHANGE: API uses camelCase now
```

## 4. 워크플로우

### 일일 작업
```bash
# 1. 최신 코드
git checkout main && git pull

# 2. 브랜치 생성
git checkout -b feature/JIRA-123-login

# 3. 작업 + 커밋 (자주, 작게)
git add .
git commit -m "feat(auth): add login form"

# 4. 푸시 + PR
git push -u origin feature/JIRA-123-login
```

### 브랜치 업데이트
```bash
# Rebase (권장)
git fetch origin
git rebase origin/main
git push -f origin feature/JIRA-123-login

# 또는 Merge
git merge origin/main
```

### PR 머지 후 정리
```bash
git checkout main && git pull
git branch -d feature/JIRA-123-login
git push origin --delete feature/JIRA-123-login
git fetch --prune
```

## 5. 금지 사항 ❌
```bash
# ❌ main 직접 커밋
git checkout main && git commit

# ❌ main force push
git push -f origin main

# ❌ 의미없는 메시지
git commit -m "fix"
git commit -m "update"

# ❌ 민감정보 커밋
git add .env

# ❌ 너무 큰 커밋 (500줄+)
git commit -m "feat: add everything"
```

## 6. 올바른 방법 ✅
```bash
# ✅ 브랜치에서 작업
git checkout -b feature/JIRA-123-add-feature

# ✅ 작은 단위 커밋
git commit -m "feat(auth): add login form"
git commit -m "feat(auth): add validation"

# ✅ 명확한 메시지
git commit -m "fix(auth): resolve token expiration"

# ✅ force push는 본인 브랜치만
git push -f origin feature/JIRA-123-my-branch
```

## 7. PR 템플릿 (.github/PULL_REQUEST_TEMPLATE.md)
```markdown
## Summary
<!-- 변경 내용 -->

## Related Issue
Closes #

## Type
- [ ] feat / fix / docs / refactor

## Checklist
- [ ] 테스트 완료
- [ ] self-review 완료
- [ ] lint/format 통과
```

## 8. 코드 리뷰 코멘트
```
🔴 [MUST] 필수 수정 - 보안 이슈
🟡 [SHOULD] 권장 - 함수 분리 제안
🟢 [COULD] 제안 - 대안 방법
❓ [Q] 질문 - 로직 의도?
👍 [NICE] 칭찬
```

## 9. Claude 실수 기록
<!-- 틀릴 때마다 추가 -->
- 커밋 메시지에서 scope 생략
- 브랜치명 티켓 번호 형식 불일치

---
*→ 명령어 상세는 git-2-commands.md 참조*
