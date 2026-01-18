package postgres

import (
	"context"

	"github.com/google/uuid"
	"gorm.io/gorm"

	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/entity"
	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/repository"
)

type battleRepository struct {
	db *gorm.DB
}

// NewBattleRepository creates a new battle repository
func NewBattleRepository(db *gorm.DB) repository.BattleRepository {
	return &battleRepository{db: db}
}

func (r *battleRepository) Create(ctx context.Context, battle *entity.Battle) error {
	return r.db.WithContext(ctx).Create(battle).Error
}

func (r *battleRepository) GetByID(ctx context.Context, id uuid.UUID) (*entity.Battle, error) {
	var battle entity.Battle
	err := r.db.WithContext(ctx).First(&battle, "id = ?", id).Error
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil
		}
		return nil, err
	}
	return &battle, nil
}

func (r *battleRepository) List(ctx context.Context, limit, offset int) ([]*entity.Battle, int64, error) {
	var battles []*entity.Battle
	var total int64

	if err := r.db.WithContext(ctx).Model(&entity.Battle{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	err := r.db.WithContext(ctx).
		Order("created_at DESC").
		Limit(limit).
		Offset(offset).
		Find(&battles).Error
	if err != nil {
		return nil, 0, err
	}

	return battles, total, nil
}

func (r *battleRepository) Update(ctx context.Context, battle *entity.Battle) error {
	return r.db.WithContext(ctx).Save(battle).Error
}

func (r *battleRepository) Delete(ctx context.Context, id uuid.UUID) error {
	return r.db.WithContext(ctx).Delete(&entity.Battle{}, "id = ?", id).Error
}

func (r *battleRepository) GetByStatus(ctx context.Context, status entity.BattleStatus, limit, offset int) ([]*entity.Battle, error) {
	var battles []*entity.Battle
	err := r.db.WithContext(ctx).
		Where("status = ?", status).
		Order("created_at DESC").
		Limit(limit).
		Offset(offset).
		Find(&battles).Error
	if err != nil {
		return nil, err
	}
	return battles, nil
}

type roundRepository struct {
	db *gorm.DB
}

// NewRoundRepository creates a new round repository
func NewRoundRepository(db *gorm.DB) repository.RoundRepository {
	return &roundRepository{db: db}
}

func (r *roundRepository) Create(ctx context.Context, round *entity.Round) error {
	return r.db.WithContext(ctx).Create(round).Error
}

func (r *roundRepository) CreateBatch(ctx context.Context, rounds []*entity.Round) error {
	return r.db.WithContext(ctx).CreateInBatches(rounds, 100).Error
}

func (r *roundRepository) GetByBattleID(ctx context.Context, battleID uuid.UUID) ([]*entity.Round, error) {
	var rounds []*entity.Round
	err := r.db.WithContext(ctx).
		Where("battle_id = ?", battleID).
		Order("round_number ASC").
		Find(&rounds).Error
	if err != nil {
		return nil, err
	}
	return rounds, nil
}

func (r *roundRepository) GetByBattleIDPaginated(ctx context.Context, battleID uuid.UUID, limit, offset int) ([]*entity.Round, int64, error) {
	var rounds []*entity.Round
	var total int64

	if err := r.db.WithContext(ctx).Model(&entity.Round{}).Where("battle_id = ?", battleID).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	err := r.db.WithContext(ctx).
		Where("battle_id = ?", battleID).
		Order("round_number ASC").
		Limit(limit).
		Offset(offset).
		Find(&rounds).Error
	if err != nil {
		return nil, 0, err
	}

	return rounds, total, nil
}

func (r *roundRepository) CountByBattleID(ctx context.Context, battleID uuid.UUID) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&entity.Round{}).Where("battle_id = ?", battleID).Count(&count).Error
	return count, err
}
