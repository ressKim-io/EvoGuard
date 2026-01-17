# 설정 관리 (Viper) & 로깅 (Zap)

## 1. Viper 설정 구조체

```go
type Config struct {
    App      AppConfig      `mapstructure:"app"`
    Database DatabaseConfig `mapstructure:"database"`
    Redis    RedisConfig    `mapstructure:"redis"`
    Log      LogConfig      `mapstructure:"log"`
}

type DatabaseConfig struct {
    Host            string        `mapstructure:"host"`
    Port            int           `mapstructure:"port"`
    MaxOpenConns    int           `mapstructure:"max_open_conns"`
    ConnMaxLifetime time.Duration `mapstructure:"conn_max_lifetime"`
    SkipDefaultTx   bool          `mapstructure:"skip_default_tx"`  // GORM 성능
    PrepareStmt     bool          `mapstructure:"prepare_stmt"`     // GORM 성능
}

type RedisConfig struct {
    Addr            string `mapstructure:"addr"`
    PoolSize        int    `mapstructure:"pool_size"`
    ReadBufferSize  int    `mapstructure:"read_buffer_size"`  // v9.12+: 32KB
    WriteBufferSize int    `mapstructure:"write_buffer_size"`
}
```

## 2. Viper 로드

```go
func Load(path string) (*Config, error) {
    viper.SetConfigFile(path)
    viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
    viper.AutomaticEnv()
    viper.SetEnvPrefix("APP")  // APP_DATABASE_HOST -> database.host
    
    setDefaults()
    if err := viper.ReadInConfig(); err != nil {
        return nil, err
    }
    
    var cfg Config
    return &cfg, viper.Unmarshal(&cfg)
}

func setDefaults() {
    viper.SetDefault("database.skip_default_tx", true)
    viper.SetDefault("database.prepare_stmt", true)
    viper.SetDefault("redis.read_buffer_size", 32768)
}
```

## 3. config.yaml 예시

```yaml
app:
  port: 8080
  environment: development

database:
  host: localhost
  port: 5432
  password: ${DB_PASSWORD}  # 환경변수 참조
  skip_default_tx: true     # GORM 성능
  prepare_stmt: true

redis:
  addr: localhost:6379
  pool_size: 10
  read_buffer_size: 32768   # 32KB
```

---

## 4. Zap 로거 설정

```go
func Init(level, format string) error {
    var cfg zap.Config
    if format == "json" {
        cfg = zap.NewProductionConfig()
    } else {
        cfg = zap.NewDevelopmentConfig()
    }
    
    cfg.EncoderConfig.TimeKey = "timestamp"
    cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
    
    log, err := cfg.Build(zap.AddStacktrace(zapcore.ErrorLevel))
    return err
}
```

## 5. Zap Best Practices

```go
// ✅ 구조화된 로깅
logger.Info("user created",
    zap.String("user_id", user.ID),
    zap.Duration("latency", time.Since(start)),
)

// ✅ 에러 컨텍스트
logger.Error("failed to create",
    zap.Error(err),
    zap.String("user_id", id),
)

// ✅ 요청별 컨텍스트
reqLogger := logger.With(zap.String("request_id", requestID))

// ❌ 문자열 포매팅 (느림)
log.Printf("user %s created", user.ID)
```

## 6. Gin 로깅 미들웨어

```go
func Logger() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        c.Next()
        
        logger.Info("request",
            zap.String("method", c.Request.Method),
            zap.String("path", c.Request.URL.Path),
            zap.Int("status", c.Writer.Status()),
            zap.Duration("latency", time.Since(start)),
        )
    }
}
```
