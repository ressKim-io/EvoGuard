# Git 지침서 (2/2) - 명령어

## 1. 기본 명령어
```bash
# 상태 확인
git status
git log --oneline -10
git branch -a

# 변경사항 확인
git diff
git diff --staged
git diff main..feature/my-branch
```

## 2. 브랜치 관리
```bash
# 생성 및 전환
git checkout -b feature/JIRA-123-login
git switch -c feature/JIRA-123-login  # Git 2.23+

# 목록
git branch        # 로컬
git branch -r     # 원격
git branch -a     # 전체

# 삭제
git branch -d feature/merged         # 머지된 브랜치
git branch -D feature/unmerged       # 강제 삭제
git push origin --delete feature/x   # 원격 삭제
```

## 3. Stash (임시 저장)
```bash
# 저장
git stash
git stash push -m "WIP: login feature"

# 목록
git stash list

# 복원
git stash pop              # 적용 후 삭제
git stash apply            # 적용 (유지)
git stash apply stash@{2}  # 특정 stash

# 삭제
git stash drop stash@{0}
git stash clear            # 전체
```

## 4. 되돌리기

### 커밋 전
```bash
git checkout -- file.txt   # 특정 파일
git restore file.txt       # Git 2.23+
git restore .              # 전체
```

### Staged 취소
```bash
git reset HEAD file.txt
git restore --staged file.txt  # Git 2.23+
```

### 커밋 수정 (push 전)
```bash
git commit --amend -m "fix: correct message"
```

### 커밋 되돌리기
```bash
# 새 커밋 생성 (안전)
git revert HEAD           # 최근 1개
git revert HEAD~3..HEAD   # 최근 3개

# 히스토리 삭제 (push 전만!)
git reset --soft HEAD~1   # 커밋만 취소 (변경사항 유지)
git reset --mixed HEAD~1  # 커밋 + staged 취소
git reset --hard HEAD~1   # 전부 삭제 (주의!)
```

## 5. Rebase

### Interactive Rebase (커밋 정리)
```bash
git rebase -i HEAD~3

# 에디터에서:
# pick   = 유지
# reword = 메시지 수정
# squash = 이전과 합치기
# drop   = 삭제
```

### main 기준 Rebase
```bash
git rebase origin/main

# 충돌 시
git add .
git rebase --continue

# 취소
git rebase --abort
```

## 6. Cherry-pick
```bash
# 특정 커밋 가져오기
git cherry-pick abc1234

# 여러 커밋
git cherry-pick abc1234 def5678

# 충돌 시
git add .
git cherry-pick --continue
```

## 7. Git Alias 설정
```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --all"

# 기본 브랜치
git config --global init.defaultBranch main

# 줄바꿈
git config --global core.autocrlf input  # Mac/Linux
git config --global core.autocrlf true   # Windows
```

## 8. .gitignore 템플릿
```gitignore
# OS
.DS_Store
Thumbs.db
*.swp

# IDE
.idea/
.vscode/
*.iml

# Dependencies
node_modules/
vendor/
__pycache__/
*.pyc
.venv/

# Build
dist/
build/
target/
*.jar
*.class

# Logs
*.log
logs/

# Environment
.env
.env.*
!.env.example

# Secrets
secrets/
*.pem
*.key
credentials.json

# Test
coverage/
.pytest_cache/

# Cache
.cache/

# Terraform
.terraform/
*.tfstate
*.tfstate.*
*.tfvars
!*.tfvars.example
```

## 9. 유용한 명령어
```bash
# 특정 파일 히스토리
git log --follow -p -- path/to/file

# 누가 수정했는지
git blame file.txt

# 커밋 검색
git log --grep="keyword"
git log --author="name"

# 파일 내용 검색
git log -S "search_string"

# 원격 정보
git remote -v
git remote show origin

# 태그
git tag v1.0.0
git push origin v1.0.0
git push origin --tags
```

## 10. 긴급 상황

### 실수로 push한 커밋 되돌리기
```bash
# revert로 새 커밋 생성 (안전)
git revert HEAD
git push

# force push (팀원 동의 필요!)
git reset --hard HEAD~1
git push -f origin feature/my-branch
```

### 잘못된 브랜치에 커밋
```bash
# 커밋 저장
git log --oneline -1  # 커밋 해시 확인

# 올바른 브랜치로 이동
git checkout correct-branch
git cherry-pick <commit-hash>

# 잘못된 브랜치에서 제거
git checkout wrong-branch
git reset --hard HEAD~1
```

### merge 충돌 해결
```bash
# 충돌 파일 확인
git status

# 수동 해결 후
git add <resolved-files>
git commit -m "merge: resolve conflicts"

# 또는 취소
git merge --abort
```

## 11. 리뷰어 체크리스트
- [ ] 코드가 PR 설명과 일치?
- [ ] 테스트 충분?
- [ ] 보안 이슈 없음?
- [ ] 성능 문제 없음?
- [ ] 코드 스타일 일관성?
- [ ] 에러 처리 적절?

---
*→ 규칙은 git-1-rules.md 참조*
