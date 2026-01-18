package entity

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

func TestNewRound(t *testing.T) {
	battleID := uuid.New()
	round := NewRound(battleID, 1, "original text", "evaded text", "leetspeak")

	assert.NotEmpty(t, round.ID)
	assert.Equal(t, battleID, round.BattleID)
	assert.Equal(t, 1, round.RoundNumber)
	assert.Equal(t, "original text", round.OriginalText)
	assert.Equal(t, "evaded text", round.EvasionText)
	assert.Equal(t, "leetspeak", round.AttackStrategy)
	assert.Equal(t, float64(0), round.ToxicScore)
	assert.Equal(t, float64(0), round.Confidence)
	assert.False(t, round.IsDetected)
}

func TestRound_SetResult(t *testing.T) {
	battleID := uuid.New()
	round := NewRound(battleID, 1, "original", "evaded", "homoglyph")

	round.SetResult(0.85, 0.92, true, 150)

	assert.Equal(t, 0.85, round.ToxicScore)
	assert.Equal(t, 0.92, round.Confidence)
	assert.True(t, round.IsDetected)
	assert.Equal(t, int64(150), round.LatencyMs)
}

func TestRound_TableName(t *testing.T) {
	round := Round{}
	assert.Equal(t, "battle_rounds", round.TableName())
}
