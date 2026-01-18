package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewBattle(t *testing.T) {
	battle := NewBattle(100, AttackStrategyMixed, "champion", true)

	assert.NotEmpty(t, battle.ID)
	assert.Equal(t, BattleStatusPending, battle.Status)
	assert.Equal(t, 100, battle.TotalRounds)
	assert.Equal(t, AttackStrategyMixed, battle.AttackStrategy)
	assert.Equal(t, "champion", battle.DefenderAlias)
	assert.True(t, battle.Async)
	assert.Equal(t, 0, battle.CompletedRounds)
	assert.Equal(t, 0, battle.EvasionCount)
	assert.Equal(t, 0, battle.DetectionCount)
}

func TestBattle_DetectionRate(t *testing.T) {
	tests := []struct {
		name            string
		completedRounds int
		detectionCount  int
		expected        float64
	}{
		{
			name:            "no rounds completed",
			completedRounds: 0,
			detectionCount:  0,
			expected:        0,
		},
		{
			name:            "all detected",
			completedRounds: 10,
			detectionCount:  10,
			expected:        1.0,
		},
		{
			name:            "half detected",
			completedRounds: 100,
			detectionCount:  50,
			expected:        0.5,
		},
		{
			name:            "70% detection",
			completedRounds: 100,
			detectionCount:  70,
			expected:        0.7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			battle := &Battle{
				CompletedRounds: tt.completedRounds,
				DetectionCount:  tt.detectionCount,
			}
			assert.Equal(t, tt.expected, battle.DetectionRate())
		})
	}
}

func TestBattle_EvasionRate(t *testing.T) {
	tests := []struct {
		name            string
		completedRounds int
		evasionCount    int
		expected        float64
	}{
		{
			name:            "no rounds completed",
			completedRounds: 0,
			evasionCount:    0,
			expected:        0,
		},
		{
			name:            "all evaded",
			completedRounds: 10,
			evasionCount:    10,
			expected:        1.0,
		},
		{
			name:            "30% evasion",
			completedRounds: 100,
			evasionCount:    30,
			expected:        0.3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			battle := &Battle{
				CompletedRounds: tt.completedRounds,
				EvasionCount:    tt.evasionCount,
			}
			assert.Equal(t, tt.expected, battle.EvasionRate())
		})
	}
}

func TestBattle_IsCompleted(t *testing.T) {
	tests := []struct {
		name     string
		status   BattleStatus
		expected bool
	}{
		{name: "pending", status: BattleStatusPending, expected: false},
		{name: "running", status: BattleStatusRunning, expected: false},
		{name: "completed", status: BattleStatusCompleted, expected: true},
		{name: "failed", status: BattleStatusFailed, expected: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			battle := &Battle{Status: tt.status}
			assert.Equal(t, tt.expected, battle.IsCompleted())
		})
	}
}

func TestBattle_CanRun(t *testing.T) {
	tests := []struct {
		name     string
		status   BattleStatus
		expected bool
	}{
		{name: "pending", status: BattleStatusPending, expected: true},
		{name: "running", status: BattleStatusRunning, expected: true},
		{name: "completed", status: BattleStatusCompleted, expected: false},
		{name: "failed", status: BattleStatusFailed, expected: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			battle := &Battle{Status: tt.status}
			assert.Equal(t, tt.expected, battle.CanRun())
		})
	}
}

func TestBattle_TableName(t *testing.T) {
	battle := Battle{}
	assert.Equal(t, "battles", battle.TableName())
}
