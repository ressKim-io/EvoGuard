package entity

import (
	"time"

	"github.com/google/uuid"
)

// Round represents a single round in a battle
type Round struct {
	ID             uuid.UUID `json:"id" gorm:"type:uuid;primary_key"`
	BattleID       uuid.UUID `json:"battle_id" gorm:"type:uuid;not null;index"`
	RoundNumber    int       `json:"round_number" gorm:"not null"`
	OriginalText   string    `json:"original_text" gorm:"type:text;not null"`
	EvasionText    string    `json:"evasion_text" gorm:"type:text;not null"`
	AttackStrategy string    `json:"attack_strategy" gorm:"type:varchar(50);not null"`
	ToxicScore     float64   `json:"toxic_score" gorm:"type:decimal(5,4)"`
	Confidence     float64   `json:"confidence" gorm:"type:decimal(5,4)"`
	IsDetected     bool      `json:"is_detected" gorm:"not null"`
	LatencyMs      int64     `json:"latency_ms" gorm:"default:0"`
	CreatedAt      time.Time `json:"created_at" gorm:"autoCreateTime"`
}

// TableName returns the table name for GORM
func (Round) TableName() string {
	return "battle_rounds"
}

// NewRound creates a new Round
func NewRound(battleID uuid.UUID, roundNumber int, originalText, evasionText, strategy string) *Round {
	return &Round{
		ID:             uuid.New(),
		BattleID:       battleID,
		RoundNumber:    roundNumber,
		OriginalText:   originalText,
		EvasionText:    evasionText,
		AttackStrategy: strategy,
	}
}

// SetResult sets the classification result for the round
func (r *Round) SetResult(toxicScore, confidence float64, isDetected bool, latencyMs int64) {
	r.ToxicScore = toxicScore
	r.Confidence = confidence
	r.IsDetected = isDetected
	r.LatencyMs = latencyMs
}
