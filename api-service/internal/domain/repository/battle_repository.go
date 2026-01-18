package repository

import (
	"context"

	"github.com/google/uuid"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/entity"
)

// BattleRepository defines the interface for battle data operations
type BattleRepository interface {
	// Create creates a new battle
	Create(ctx context.Context, battle *entity.Battle) error

	// GetByID retrieves a battle by its ID
	GetByID(ctx context.Context, id uuid.UUID) (*entity.Battle, error)

	// List retrieves battles with pagination
	List(ctx context.Context, limit, offset int) ([]*entity.Battle, int64, error)

	// Update updates a battle
	Update(ctx context.Context, battle *entity.Battle) error

	// Delete deletes a battle by ID
	Delete(ctx context.Context, id uuid.UUID) error

	// GetByStatus retrieves battles by status
	GetByStatus(ctx context.Context, status entity.BattleStatus, limit, offset int) ([]*entity.Battle, error)
}

// RoundRepository defines the interface for round data operations
type RoundRepository interface {
	// Create creates a new round
	Create(ctx context.Context, round *entity.Round) error

	// CreateBatch creates multiple rounds at once
	CreateBatch(ctx context.Context, rounds []*entity.Round) error

	// GetByBattleID retrieves all rounds for a battle
	GetByBattleID(ctx context.Context, battleID uuid.UUID) ([]*entity.Round, error)

	// GetByBattleIDPaginated retrieves rounds with pagination
	GetByBattleIDPaginated(ctx context.Context, battleID uuid.UUID, limit, offset int) ([]*entity.Round, int64, error)

	// CountByBattleID counts rounds for a battle
	CountByBattleID(ctx context.Context, battleID uuid.UUID) (int64, error)
}
