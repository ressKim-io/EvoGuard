package entity

import (
	"time"

	"github.com/google/uuid"
)

// BattleStatus represents the current state of a battle
type BattleStatus string

const (
	BattleStatusPending   BattleStatus = "pending"
	BattleStatusRunning   BattleStatus = "running"
	BattleStatusCompleted BattleStatus = "completed"
	BattleStatusFailed    BattleStatus = "failed"
)

// AttackStrategy represents the attack strategy type
type AttackStrategy string

const (
	AttackStrategyUnicodeEvasion AttackStrategy = "unicode_evasion"
	AttackStrategyHomoglyph      AttackStrategy = "homoglyph"
	AttackStrategyLeetspeak      AttackStrategy = "leetspeak"
	AttackStrategyLLMEvasion     AttackStrategy = "llm_evasion"
	AttackStrategyAdversarialLLM AttackStrategy = "adversarial_llm"
	AttackStrategyMixed          AttackStrategy = "mixed"
)

// Battle represents a battle between attacker and defender
type Battle struct {
	ID              uuid.UUID      `json:"id" gorm:"type:uuid;primary_key"`
	Status          BattleStatus   `json:"status" gorm:"type:varchar(20);not null;default:'pending'"`
	TotalRounds     int            `json:"total_rounds" gorm:"not null"`
	CompletedRounds int            `json:"completed_rounds" gorm:"default:0"`
	AttackStrategy  AttackStrategy `json:"attack_strategy" gorm:"type:varchar(50);not null"`
	DefenderAlias   string         `json:"defender_alias" gorm:"type:varchar(50);not null;default:'champion'"`
	EvasionCount    int            `json:"evasion_count" gorm:"default:0"`
	DetectionCount  int            `json:"detection_count" gorm:"default:0"`
	Async           bool           `json:"async" gorm:"default:true"`
	CreatedAt       time.Time      `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt       time.Time      `json:"updated_at" gorm:"autoUpdateTime"`

	// Relations
	Rounds []Round `json:"rounds,omitempty" gorm:"foreignKey:BattleID"`
}

// TableName returns the table name for GORM
func (Battle) TableName() string {
	return "battles"
}

// NewBattle creates a new Battle with default values
func NewBattle(totalRounds int, strategy AttackStrategy, defenderAlias string, async bool) *Battle {
	return &Battle{
		ID:             uuid.New(),
		Status:         BattleStatusPending,
		TotalRounds:    totalRounds,
		AttackStrategy: strategy,
		DefenderAlias:  defenderAlias,
		Async:          async,
	}
}

// DetectionRate returns the current detection rate
func (b *Battle) DetectionRate() float64 {
	if b.CompletedRounds == 0 {
		return 0
	}
	return float64(b.DetectionCount) / float64(b.CompletedRounds)
}

// EvasionRate returns the current evasion rate
func (b *Battle) EvasionRate() float64 {
	if b.CompletedRounds == 0 {
		return 0
	}
	return float64(b.EvasionCount) / float64(b.CompletedRounds)
}

// IsCompleted returns true if the battle is completed
func (b *Battle) IsCompleted() bool {
	return b.Status == BattleStatusCompleted
}

// CanRun returns true if the battle can be run
func (b *Battle) CanRun() bool {
	return b.Status == BattleStatusPending || b.Status == BattleStatusRunning
}
