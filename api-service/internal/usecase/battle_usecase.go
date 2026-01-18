package usecase

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/entity"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/repository"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/service"
)

// Error definitions for battle usecase
var (
	ErrBattleNotFound    = errors.New("battle not found")
	ErrInvalidRequest    = errors.New("invalid request")
	ErrBattleNotRunnable = errors.New("battle cannot accept rounds")
	ErrBattleCompleted   = errors.New("battle already completed")
)

// CreateBattleInput represents the input for creating a battle
type CreateBattleInput struct {
	Rounds         int    `json:"rounds" binding:"required,min=1,max=10000"`
	AttackStrategy string `json:"attack_strategy" binding:"required"`
	DefenderAlias  string `json:"defender_alias"`
	Async          bool   `json:"async"`
}

// BattleOutput represents the output for battle operations
type BattleOutput struct {
	BattleID        uuid.UUID `json:"battle_id"`
	Status          string    `json:"status"`
	TotalRounds     int       `json:"total_rounds"`
	CompletedRounds int       `json:"completed_rounds"`
	DetectionRate   float64   `json:"detection_rate"`
	EvasionRate     float64   `json:"evasion_rate"`
	AttackStrategy  string    `json:"attack_strategy"`
	DefenderAlias   string    `json:"defender_alias"`
	CreatedAt       string    `json:"created_at"`
}

// BattleListOutput represents paginated battle list
type BattleListOutput struct {
	Battles []*BattleOutput `json:"battles"`
	Total   int64           `json:"total"`
	Limit   int             `json:"limit"`
	Offset  int             `json:"offset"`
	HasMore bool            `json:"has_more"`
}

// SubmitRoundInput represents the input for submitting a round
type SubmitRoundInput struct {
	OriginalText   string `json:"original_text" binding:"required"`
	EvasionText    string `json:"evasion_text" binding:"required"`
	AttackStrategy string `json:"attack_strategy" binding:"required"`
}

// RoundOutput represents the output for a round
type RoundOutput struct {
	RoundID        uuid.UUID `json:"round_id"`
	BattleID       uuid.UUID `json:"battle_id"`
	RoundNumber    int       `json:"round_number"`
	OriginalText   string    `json:"original_text"`
	EvasionText    string    `json:"evasion_text"`
	AttackStrategy string    `json:"attack_strategy"`
	ToxicScore     float64   `json:"toxic_score"`
	Confidence     float64   `json:"confidence"`
	IsDetected     bool      `json:"is_detected"`
	LatencyMs      int64     `json:"latency_ms"`
}

// BattleUsecase defines the interface for battle business logic
type BattleUsecase interface {
	Create(ctx context.Context, input *CreateBattleInput) (*BattleOutput, error)
	GetByID(ctx context.Context, id uuid.UUID) (*BattleOutput, error)
	List(ctx context.Context, limit, offset int) (*BattleListOutput, error)
	Stop(ctx context.Context, id uuid.UUID) (*BattleOutput, error)
	SubmitRound(ctx context.Context, battleID uuid.UUID, input *SubmitRoundInput) (*RoundOutput, error)
	GetRounds(ctx context.Context, battleID uuid.UUID, limit, offset int) ([]*RoundOutput, int64, error)
}

type battleUsecase struct {
	battleRepo repository.BattleRepository
	roundRepo  repository.RoundRepository
	classifier service.Classifier
}

// NewBattleUsecase creates a new battle usecase
func NewBattleUsecase(battleRepo repository.BattleRepository, roundRepo repository.RoundRepository, classifier service.Classifier) BattleUsecase {
	return &battleUsecase{
		battleRepo: battleRepo,
		roundRepo:  roundRepo,
		classifier: classifier,
	}
}

func (u *battleUsecase) Create(ctx context.Context, input *CreateBattleInput) (*BattleOutput, error) {
	strategy := entity.AttackStrategy(input.AttackStrategy)
	defenderAlias := input.DefenderAlias
	if defenderAlias == "" {
		defenderAlias = "champion"
	}

	battle := entity.NewBattle(input.Rounds, strategy, defenderAlias, input.Async)

	if err := u.battleRepo.Create(ctx, battle); err != nil {
		return nil, err
	}

	return toBattleOutput(battle), nil
}

func (u *battleUsecase) GetByID(ctx context.Context, id uuid.UUID) (*BattleOutput, error) {
	battle, err := u.battleRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if battle == nil {
		return nil, ErrBattleNotFound
	}

	return toBattleOutput(battle), nil
}

func (u *battleUsecase) List(ctx context.Context, limit, offset int) (*BattleListOutput, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	battles, total, err := u.battleRepo.List(ctx, limit, offset)
	if err != nil {
		return nil, err
	}

	outputs := make([]*BattleOutput, len(battles))
	for i, b := range battles {
		outputs[i] = toBattleOutput(b)
	}

	return &BattleListOutput{
		Battles: outputs,
		Total:   total,
		Limit:   limit,
		Offset:  offset,
		HasMore: int64(offset+limit) < total,
	}, nil
}

func (u *battleUsecase) Stop(ctx context.Context, id uuid.UUID) (*BattleOutput, error) {
	battle, err := u.battleRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if battle == nil {
		return nil, ErrBattleNotFound
	}

	if battle.Status == entity.BattleStatusCompleted || battle.Status == entity.BattleStatusFailed {
		return toBattleOutput(battle), nil
	}

	battle.Status = entity.BattleStatusCompleted
	if err := u.battleRepo.Update(ctx, battle); err != nil {
		return nil, err
	}

	return toBattleOutput(battle), nil
}

func (u *battleUsecase) SubmitRound(ctx context.Context, battleID uuid.UUID, input *SubmitRoundInput) (*RoundOutput, error) {
	battle, err := u.battleRepo.GetByID(ctx, battleID)
	if err != nil {
		return nil, err
	}
	if battle == nil {
		return nil, ErrBattleNotFound
	}

	if !battle.CanRun() {
		return nil, ErrBattleNotRunnable
	}

	if battle.CompletedRounds >= battle.TotalRounds {
		return nil, ErrBattleCompleted
	}

	// Update battle status to running if pending
	if battle.Status == entity.BattleStatusPending {
		battle.Status = entity.BattleStatusRunning
	}

	// Create round
	roundNumber := battle.CompletedRounds + 1
	round := entity.NewRound(battleID, roundNumber, input.OriginalText, input.EvasionText, input.AttackStrategy)

	// Classify the evasion text using ML service
	start := time.Now()
	var toxicScore, confidence float64
	var isDetected bool

	if u.classifier != nil {
		result, err := u.classifier.Classify(ctx, input.EvasionText, round.ID.String())
		if err != nil {
			return nil, err
		}
		toxicScore = result.Score
		confidence = result.Confidence
		isDetected = result.Label == "toxic"
	}
	latencyMs := time.Since(start).Milliseconds()

	// Set round result
	round.SetResult(toxicScore, confidence, isDetected, latencyMs)

	// Save round
	if err := u.roundRepo.Create(ctx, round); err != nil {
		return nil, err
	}

	// Update battle stats
	battle.CompletedRounds++
	if isDetected {
		battle.DetectionCount++
	} else {
		battle.EvasionCount++
	}

	// Mark battle as completed if all rounds done
	if battle.CompletedRounds >= battle.TotalRounds {
		battle.Status = entity.BattleStatusCompleted
	}

	if err := u.battleRepo.Update(ctx, battle); err != nil {
		return nil, err
	}

	return toRoundOutput(round), nil
}

func (u *battleUsecase) GetRounds(ctx context.Context, battleID uuid.UUID, limit, offset int) ([]*RoundOutput, int64, error) {
	battle, err := u.battleRepo.GetByID(ctx, battleID)
	if err != nil {
		return nil, 0, err
	}
	if battle == nil {
		return nil, 0, ErrBattleNotFound
	}

	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	rounds, total, err := u.roundRepo.GetByBattleIDPaginated(ctx, battleID, limit, offset)
	if err != nil {
		return nil, 0, err
	}

	outputs := make([]*RoundOutput, len(rounds))
	for i, r := range rounds {
		outputs[i] = toRoundOutput(r)
	}

	return outputs, total, nil
}

func toBattleOutput(b *entity.Battle) *BattleOutput {
	return &BattleOutput{
		BattleID:        b.ID,
		Status:          string(b.Status),
		TotalRounds:     b.TotalRounds,
		CompletedRounds: b.CompletedRounds,
		DetectionRate:   b.DetectionRate(),
		EvasionRate:     b.EvasionRate(),
		AttackStrategy:  string(b.AttackStrategy),
		DefenderAlias:   b.DefenderAlias,
		CreatedAt:       b.CreatedAt.Format("2006-01-02T15:04:05Z"),
	}
}

func toRoundOutput(r *entity.Round) *RoundOutput {
	return &RoundOutput{
		RoundID:        r.ID,
		BattleID:       r.BattleID,
		RoundNumber:    r.RoundNumber,
		OriginalText:   r.OriginalText,
		EvasionText:    r.EvasionText,
		AttackStrategy: r.AttackStrategy,
		ToxicScore:     r.ToxicScore,
		Confidence:     r.Confidence,
		IsDetected:     r.IsDetected,
		LatencyMs:      r.LatencyMs,
	}
}
