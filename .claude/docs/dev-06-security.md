# 보안 가이드

> 시크릿 관리 및 보안 Best Practices

## 개요

EvoGuard 프로젝트의 보안 원칙:
1. **최소 권한 원칙 (PoLP)**: 필요한 권한만 부여
2. **시크릿 분리**: 코드와 시크릿 완전 분리
3. **암호화**: 저장 및 전송 시 암호화
4. **감사**: 접근 로그 및 모니터링

## 시크릿 관리

### 환경별 시크릿 저장소

| 환경 | 저장소 | 설명 |
|------|--------|------|
| 개발 | `.env` 파일 | 로컬 전용, gitignore |
| CI/CD | GitHub Secrets | 워크플로우 전용 |
| Staging | K8s Secrets | 암호화 권장 |
| Production | Vault / AWS Secrets | 필수 암호화 |

### 절대 커밋하면 안 되는 것

```gitignore
# 환경 변수
.env
.env.local
.env.*.local

# 인증서/키
*.pem
*.key
*.p12
*.pfx

# 자격 증명
credentials.json
service-account.json
kubeconfig
```

### .env 파일 관리

```bash
# 1. 템플릿에서 복사
cp .env.example .env

# 2. 실제 값 입력
vim .env

# 3. 절대 커밋하지 않음 (자동으로 gitignore됨)
```

## Kubernetes 시크릿

### 기본 시크릿 생성

```bash
# 리터럴 값
kubectl create secret generic db-credentials \
  --from-literal=username=postgres \
  --from-literal=password=your-password

# 파일에서
kubectl create secret generic tls-certs \
  --from-file=tls.crt=./server.crt \
  --from-file=tls.key=./server.key
```

### YAML로 시크릿 정의

```yaml
# k8s/secrets/db-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
  namespace: evoguard
type: Opaque
stringData:  # 자동으로 base64 인코딩
  DB_USER: postgres
  DB_PASSWORD: "${DB_PASSWORD}"  # 실제 값은 CI/CD에서 주입
```

### Sealed Secrets (권장)

```bash
# kubeseal 설치
brew install kubeseal

# 시크릿 암호화
kubeseal --format yaml < secret.yaml > sealed-secret.yaml

# 암호화된 시크릿은 Git에 커밋 가능
git add sealed-secret.yaml
```

### External Secrets Operator

```yaml
# AWS Secrets Manager 연동
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets
    kind: ClusterSecretStore
  target:
    name: db-credentials
  data:
    - secretKey: DB_PASSWORD
      remoteRef:
        key: evoguard/db
        property: password
```

## API 인증

### JWT 설정

```go
// Go API - JWT 검증
type JWTConfig struct {
    Secret     string        `mapstructure:"secret"`
    Expiry     time.Duration `mapstructure:"expiry"`
    Issuer     string        `mapstructure:"issuer"`
}

// 환경 변수에서 로드
// JWT_SECRET=your-256-bit-secret
// JWT_EXPIRY=24h
```

### API Key 관리

```python
# Python ML Service - API Key 검증
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("ML_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

## RBAC (Kubernetes)

### 서비스 계정 생성

```yaml
# k8s/rbac/service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: evoguard-api
  namespace: evoguard
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: evoguard-api-role
  namespace: evoguard
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["db-credentials"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: evoguard-api-binding
  namespace: evoguard
subjects:
  - kind: ServiceAccount
    name: evoguard-api
roleRef:
  kind: Role
  name: evoguard-api-role
  apiGroup: rbac.authorization.k8s.io
```

## 보안 스캐닝

### 코드 스캐닝

```bash
# Go - gosec
gosec ./...

# Python - bandit
bandit -r ml-service/

# 컨테이너 - trivy
trivy image ghcr.io/your-org/evoguard/api-service:latest
```

### CI에서 자동 스캔

```yaml
# .github/workflows/security.yml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ env.IMAGE }}
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload Trivy scan results
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: 'trivy-results.sarif'
```

## 네트워크 보안

### Network Policy (K8s)

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-service-policy
  namespace: evoguard
spec:
  podSelector:
    matchLabels:
      app: api-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ingress-nginx
      ports:
        - port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - port: 5432
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
```

## 시크릿 로테이션

### 자동 로테이션 설정

```bash
# AWS Secrets Manager 로테이션
aws secretsmanager rotate-secret \
  --secret-id evoguard/db \
  --rotation-rules AutomaticallyAfterDays=30

# Vault 로테이션
vault write database/rotate-root/evoguard-db
```

### 로테이션 후 재배포

```bash
# K8s 시크릿 업데이트 후 롤아웃
kubectl rollout restart deployment/api-service -n evoguard
```

## 체크리스트

### 개발 환경
- [ ] `.env`가 `.gitignore`에 포함
- [ ] 민감 정보 하드코딩 없음
- [ ] API 키/토큰 환경 변수 사용

### CI/CD
- [ ] GitHub Secrets 사용
- [ ] 시크릿이 로그에 노출되지 않음
- [ ] 보안 스캔 통과

### Production
- [ ] 시크릿 암호화 (at rest)
- [ ] TLS 적용
- [ ] RBAC 최소 권한
- [ ] 네트워크 정책 적용
- [ ] 감사 로깅 활성화

## 참고 자료

- [Kubernetes Secrets Best Practices](https://kubernetes.io/docs/concepts/security/secrets-good-practices/)
- [CNCF Secrets Management](https://www.cncf.io/blog/2023/09/28/kubernetes-security-best-practices-for-kubernetes-secrets-management/)
- [GitGuardian - K8s Secrets](https://blog.gitguardian.com/how-to-handle-secrets-in-kubernetes/)
- [HashiCorp Vault](https://www.vaultproject.io/)

---

*관련 문서: `03-ENVIRONMENT_SETUP.md`, `dev-02-environment.md`*
