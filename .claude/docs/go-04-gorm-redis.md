# GORM & go-redis Best Practices

## 1. GORM 초기화 (성능 최적화)

```go
db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
    SkipDefaultTransaction: true,  // 매 쓰기마다 트랜잭션 X
    PrepareStmt:            true,  // Prepared Statement 캐싱
})

sqlDB, _ := db.DB()
sqlDB.SetMaxIdleConns(10)
sqlDB.SetMaxOpenConns(100)
sqlDB.SetConnMaxLifetime(time.Hour)
```

## 2. GORM 쿼리 패턴

### Context 필수
```go
func (r *UserRepo) FindByID(ctx context.Context, id string) (*User, error) {
    var user User
    return &user, r.db.WithContext(ctx).First(&user, "id = ?", id).Error
}
```

### N+1 방지 (Preload)
```go
// ❌ N+1 발생
users := []User{}
db.Find(&users)
for _, u := range users {
    db.Model(&u).Association("Orders").Find(&u.Orders)
}

// ✅ Preload
db.Preload("Orders").Preload("Orders.Items").Find(&users)
```

### 필요한 필드만 조회
```go
db.Select("id", "name", "email").First(&user, id)
```

### 배치 처리
```go
db.FindInBatches(&users, 100, func(tx *gorm.DB, batch int) error {
    for _, u := range users { /* 처리 */ }
    return nil
})
```

### 트랜잭션
```go
db.Transaction(func(tx *gorm.DB) error {
    if err := tx.Create(&user).Error; err != nil {
        return err  // 자동 롤백
    }
    return tx.Create(&profile).Error
})
```

### Scope 재사용
```go
func Active(db *gorm.DB) *gorm.DB {
    return db.Where("status = ?", "active")
}
db.Scopes(Active).Find(&users)
```

---

## 3. go-redis 초기화 (v9.7.0)

```go
client := redis.NewClient(&redis.Options{
    Addr:            "localhost:6379",
    PoolSize:        10,
    MinIdleConns:    5,
    DialTimeout:     5 * time.Second,
    ReadTimeout:     3 * time.Second,
    WriteTimeout:    3 * time.Second,
    Protocol:        3,                 // RESP3
    ReadBufferSize:  32768,             // 32KB (v9.12+)
    WriteBufferSize: 32768,
})
```

## 4. Redis 패턴

### Pipeline (배치)
```go
pipe := client.Pipeline()
for _, key := range keys {
    pipe.Get(ctx, key)
}
cmds, _ := pipe.Exec(ctx)
```

### Cache-Aside
```go
func GetOrSet(ctx context.Context, key string, ttl time.Duration, fn func() (string, error)) (string, error) {
    val, err := client.Get(ctx, key).Result()
    if err == nil {
        return val, nil
    }
    if err != redis.Nil {
        return "", err
    }
    
    val, err = fn()  // 원본 조회
    if err != nil {
        return "", err
    }
    
    _ = client.Set(ctx, key, val, ttl).Err()
    return val, nil
}
```

### 분산 락
```go
func AcquireLock(ctx context.Context, key string, ttl time.Duration) (bool, error) {
    return client.SetNX(ctx, "lock:"+key, "1", ttl).Result()
}
```

## 5. 에러 처리

```go
// GORM
if errors.Is(err, gorm.ErrRecordNotFound) {
    return nil, ErrUserNotFound  // 도메인 에러로 변환
}

// Redis
if err == redis.Nil {
    return "", ErrCacheMiss
}
```
