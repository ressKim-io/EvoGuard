package usecase

import (
	"context"
	"errors"

	"github.com/google/uuid"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/entity"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/repository"
)

// Error definitions for battle usecase
var (
	ErrBattleNotFound = errors.New("battle not found")
	ErrInvalidRequest = errors.New("invalid request")
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

// BattleUsecase defines the interface for battle business logic
type BattleUsecase interface {
	Create(ctx context.Context, input *CreateBattleInput) (*BattleOutput, error)
	GetByID(ctx context.Context, id uuid.UUID) (*BattleOutput, error)
	List(ctx context.Context, limit, offset int) (*BattleListOutput, error)
	Stop(ctx context.Context, id uuid.UUID) (*BattleOutput, error)
}

type battleUsecase struct {
	battleRepo repository.BattleRepository
	roundRepo  repository.RoundRepository
}

// NewBattleUsecase creates a new battle usecase
func NewBattleUsecase(battleRepo repository.BattleRepository, roundRepo repository.RoundRepository) BattleUsecase {
	return &battleUsecase{
		battleRepo: battleRepo,
		roundRepo:  roundRepo,
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
