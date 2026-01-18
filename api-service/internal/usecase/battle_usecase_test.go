package usecase

import (
	"context"
	"errors"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/ressKim-io/EvoGuard/api-service/internal/domain/entity"
)

// MockBattleRepository is a mock implementation of BattleRepository
type MockBattleRepository struct {
	mock.Mock
}

func (m *MockBattleRepository) Create(ctx context.Context, battle *entity.Battle) error {
	args := m.Called(ctx, battle)
	return args.Error(0)
}

func (m *MockBattleRepository) GetByID(ctx context.Context, id uuid.UUID) (*entity.Battle, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*entity.Battle), args.Error(1)
}

func (m *MockBattleRepository) List(ctx context.Context, limit, offset int) ([]*entity.Battle, int64, error) {
	args := m.Called(ctx, limit, offset)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*entity.Battle), args.Get(1).(int64), args.Error(2)
}

func (m *MockBattleRepository) Update(ctx context.Context, battle *entity.Battle) error {
	args := m.Called(ctx, battle)
	return args.Error(0)
}

func (m *MockBattleRepository) Delete(ctx context.Context, id uuid.UUID) error {
	args := m.Called(ctx, id)
	return args.Error(0)
}

func (m *MockBattleRepository) GetByStatus(ctx context.Context, status entity.BattleStatus, limit, offset int) ([]*entity.Battle, error) {
	args := m.Called(ctx, status, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*entity.Battle), args.Error(1)
}

// MockRoundRepository is a mock implementation of RoundRepository
type MockRoundRepository struct {
	mock.Mock
}

func (m *MockRoundRepository) Create(ctx context.Context, round *entity.Round) error {
	args := m.Called(ctx, round)
	return args.Error(0)
}

func (m *MockRoundRepository) CreateBatch(ctx context.Context, rounds []*entity.Round) error {
	args := m.Called(ctx, rounds)
	return args.Error(0)
}

func (m *MockRoundRepository) GetByBattleID(ctx context.Context, battleID uuid.UUID) ([]*entity.Round, error) {
	args := m.Called(ctx, battleID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*entity.Round), args.Error(1)
}

func (m *MockRoundRepository) GetByBattleIDPaginated(ctx context.Context, battleID uuid.UUID, limit, offset int) ([]*entity.Round, int64, error) {
	args := m.Called(ctx, battleID, limit, offset)
	if args.Get(0) == nil {
		return nil, 0, args.Error(2)
	}
	return args.Get(0).([]*entity.Round), args.Get(1).(int64), args.Error(2)
}

func (m *MockRoundRepository) CountByBattleID(ctx context.Context, battleID uuid.UUID) (int64, error) {
	args := m.Called(ctx, battleID)
	return args.Get(0).(int64), args.Error(1)
}

func TestBattleUsecase_Create(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		mockBattleRepo.On("Create", mock.Anything, mock.AnythingOfType("*entity.Battle")).Return(nil)

		input := &CreateBattleInput{
			Rounds:         100,
			AttackStrategy: "mixed",
			DefenderAlias:  "champion",
			Async:          true,
		}

		output, err := uc.Create(context.Background(), input)

		assert.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, "pending", output.Status)
		assert.Equal(t, 100, output.TotalRounds)
		assert.Equal(t, "mixed", output.AttackStrategy)
		assert.Equal(t, "champion", output.DefenderAlias)
		mockBattleRepo.AssertExpectations(t)
	})

	t.Run("success with default defender alias", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		mockBattleRepo.On("Create", mock.Anything, mock.AnythingOfType("*entity.Battle")).Return(nil)

		input := &CreateBattleInput{
			Rounds:         50,
			AttackStrategy: "leetspeak",
			Async:          false,
		}

		output, err := uc.Create(context.Background(), input)

		assert.NoError(t, err)
		assert.Equal(t, "champion", output.DefenderAlias)
	})

	t.Run("repository error", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		expectedErr := errors.New("database error")
		mockBattleRepo.On("Create", mock.Anything, mock.AnythingOfType("*entity.Battle")).Return(expectedErr)

		input := &CreateBattleInput{
			Rounds:         100,
			AttackStrategy: "mixed",
		}

		output, err := uc.Create(context.Background(), input)

		assert.Error(t, err)
		assert.Nil(t, output)
		assert.Equal(t, expectedErr, err)
	})
}

func TestBattleUsecase_GetByID(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		battle := &entity.Battle{
			ID:              battleID,
			Status:          entity.BattleStatusRunning,
			TotalRounds:     100,
			CompletedRounds: 50,
			EvasionCount:    15,
			DetectionCount:  35,
			AttackStrategy:  entity.AttackStrategyMixed,
			DefenderAlias:   "champion",
		}

		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(battle, nil)

		output, err := uc.GetByID(context.Background(), battleID)

		assert.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, battleID, output.BattleID)
		assert.Equal(t, "running", output.Status)
		assert.Equal(t, 0.7, output.DetectionRate)
		assert.Equal(t, 0.3, output.EvasionRate)
	})

	t.Run("not found", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(nil, nil)

		output, err := uc.GetByID(context.Background(), battleID)

		assert.Error(t, err)
		assert.Equal(t, ErrBattleNotFound, err)
		assert.Nil(t, output)
	})

	t.Run("repository error", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		expectedErr := errors.New("database error")
		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(nil, expectedErr)

		output, err := uc.GetByID(context.Background(), battleID)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Nil(t, output)
	})
}

func TestBattleUsecase_List(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battles := []*entity.Battle{
			{ID: uuid.New(), Status: entity.BattleStatusCompleted, TotalRounds: 100},
			{ID: uuid.New(), Status: entity.BattleStatusRunning, TotalRounds: 50},
		}

		mockBattleRepo.On("List", mock.Anything, 20, 0).Return(battles, int64(2), nil)

		output, err := uc.List(context.Background(), 20, 0)

		assert.NoError(t, err)
		assert.NotNil(t, output)
		assert.Len(t, output.Battles, 2)
		assert.Equal(t, int64(2), output.Total)
		assert.Equal(t, 20, output.Limit)
		assert.Equal(t, 0, output.Offset)
		assert.False(t, output.HasMore)
	})

	t.Run("with pagination - has more", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battles := []*entity.Battle{
			{ID: uuid.New(), Status: entity.BattleStatusCompleted},
		}

		mockBattleRepo.On("List", mock.Anything, 10, 0).Return(battles, int64(50), nil)

		output, err := uc.List(context.Background(), 10, 0)

		assert.NoError(t, err)
		assert.True(t, output.HasMore)
	})

	t.Run("default limit when zero", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		mockBattleRepo.On("List", mock.Anything, 20, 0).Return([]*entity.Battle{}, int64(0), nil)

		output, err := uc.List(context.Background(), 0, 0)

		assert.NoError(t, err)
		assert.Equal(t, 20, output.Limit)
	})

	t.Run("cap limit at 100", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		mockBattleRepo.On("List", mock.Anything, 100, 0).Return([]*entity.Battle{}, int64(0), nil)

		output, err := uc.List(context.Background(), 500, 0)

		assert.NoError(t, err)
		assert.Equal(t, 100, output.Limit)
	})

	t.Run("repository error", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		expectedErr := errors.New("database error")
		mockBattleRepo.On("List", mock.Anything, 20, 0).Return(nil, int64(0), expectedErr)

		output, err := uc.List(context.Background(), 20, 0)

		assert.Error(t, err)
		assert.Nil(t, output)
	})
}

func TestBattleUsecase_Stop(t *testing.T) {
	t.Run("success - running battle", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		battle := &entity.Battle{
			ID:     battleID,
			Status: entity.BattleStatusRunning,
		}

		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(battle, nil)
		mockBattleRepo.On("Update", mock.Anything, mock.AnythingOfType("*entity.Battle")).Return(nil)

		output, err := uc.Stop(context.Background(), battleID)

		assert.NoError(t, err)
		assert.Equal(t, "completed", output.Status)
	})

	t.Run("already completed", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		battle := &entity.Battle{
			ID:     battleID,
			Status: entity.BattleStatusCompleted,
		}

		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(battle, nil)

		output, err := uc.Stop(context.Background(), battleID)

		assert.NoError(t, err)
		assert.Equal(t, "completed", output.Status)
		// Update should not be called
		mockBattleRepo.AssertNotCalled(t, "Update", mock.Anything, mock.Anything)
	})

	t.Run("already failed", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		battle := &entity.Battle{
			ID:     battleID,
			Status: entity.BattleStatusFailed,
		}

		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(battle, nil)

		output, err := uc.Stop(context.Background(), battleID)

		assert.NoError(t, err)
		assert.Equal(t, "failed", output.Status)
	})

	t.Run("not found", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(nil, nil)

		output, err := uc.Stop(context.Background(), battleID)

		assert.Error(t, err)
		assert.Equal(t, ErrBattleNotFound, err)
		assert.Nil(t, output)
	})

	t.Run("update error", func(t *testing.T) {
		mockBattleRepo := new(MockBattleRepository)
		mockRoundRepo := new(MockRoundRepository)
		uc := NewBattleUsecase(mockBattleRepo, mockRoundRepo, nil)

		battleID := uuid.New()
		battle := &entity.Battle{
			ID:     battleID,
			Status: entity.BattleStatusRunning,
		}

		expectedErr := errors.New("update error")
		mockBattleRepo.On("GetByID", mock.Anything, battleID).Return(battle, nil)
		mockBattleRepo.On("Update", mock.Anything, mock.AnythingOfType("*entity.Battle")).Return(expectedErr)

		output, err := uc.Stop(context.Background(), battleID)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Nil(t, output)
	})
}

func TestToBattleOutput(t *testing.T) {
	battle := &entity.Battle{
		ID:              uuid.New(),
		Status:          entity.BattleStatusRunning,
		TotalRounds:     100,
		CompletedRounds: 75,
		EvasionCount:    20,
		DetectionCount:  55,
		AttackStrategy:  entity.AttackStrategyLeetspeak,
		DefenderAlias:   "challenger",
	}

	output := toBattleOutput(battle)

	assert.Equal(t, battle.ID, output.BattleID)
	assert.Equal(t, "running", output.Status)
	assert.Equal(t, 100, output.TotalRounds)
	assert.Equal(t, 75, output.CompletedRounds)
	assert.InDelta(t, 0.733, output.DetectionRate, 0.01)
	assert.InDelta(t, 0.266, output.EvasionRate, 0.01)
	assert.Equal(t, "leetspeak", output.AttackStrategy)
	assert.Equal(t, "challenger", output.DefenderAlias)
}
