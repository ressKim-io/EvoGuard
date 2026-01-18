package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/ressKim-io/EvoGuard/api-service/internal/usecase"
)

// MockBattleUsecase is a mock implementation of BattleUsecase
type MockBattleUsecase struct {
	mock.Mock
}

func (m *MockBattleUsecase) Create(ctx context.Context, input *usecase.CreateBattleInput) (*usecase.BattleOutput, error) {
	args := m.Called(ctx, input)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*usecase.BattleOutput), args.Error(1)
}

func (m *MockBattleUsecase) GetByID(ctx context.Context, id uuid.UUID) (*usecase.BattleOutput, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*usecase.BattleOutput), args.Error(1)
}

func (m *MockBattleUsecase) List(ctx context.Context, limit, offset int) (*usecase.BattleListOutput, error) {
	args := m.Called(ctx, limit, offset)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*usecase.BattleListOutput), args.Error(1)
}

func (m *MockBattleUsecase) Stop(ctx context.Context, id uuid.UUID) (*usecase.BattleOutput, error) {
	args := m.Called(ctx, id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*usecase.BattleOutput), args.Error(1)
}

func setupTestRouter(h *BattleHandler) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/api/v1/battles", h.CreateBattle)
	r.GET("/api/v1/battles", h.ListBattles)
	r.GET("/api/v1/battles/:id", h.GetBattle)
	r.GET("/api/v1/battles/:id/stats", h.GetBattleStats)
	r.POST("/api/v1/battles/:id/stop", h.StopBattle)
	return r
}

func TestCreateBattle_Success(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	expectedOutput := &usecase.BattleOutput{
		BattleID:        battleID,
		Status:          "pending",
		TotalRounds:     100,
		CompletedRounds: 0,
		DetectionRate:   0,
		EvasionRate:     0,
		AttackStrategy:  "mixed",
		DefenderAlias:   "champion",
		CreatedAt:       "2026-01-18T12:00:00Z",
	}

	mockUC.On("Create", mock.Anything, mock.MatchedBy(func(input *usecase.CreateBattleInput) bool {
		return input.Rounds == 100 && input.AttackStrategy == "mixed"
	})).Return(expectedOutput, nil)

	body := `{"rounds": 100, "attack_strategy": "mixed", "async": true}`
	req, _ := http.NewRequest("POST", "/api/v1/battles", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusCreated, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	mockUC.AssertExpectations(t)
}

func TestCreateBattle_InvalidStrategy(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	body := `{"rounds": 100, "attack_strategy": "invalid_strategy", "async": true}`
	req, _ := http.NewRequest("POST", "/api/v1/battles", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.False(t, response.Success)
	assert.Equal(t, "INVALID_REQUEST", response.Error.Code)
}

func TestCreateBattle_InvalidJSON(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	body := `{"rounds": "invalid"}`
	req, _ := http.NewRequest("POST", "/api/v1/battles", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
}

func TestGetBattle_Success(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	expectedOutput := &usecase.BattleOutput{
		BattleID:        battleID,
		Status:          "running",
		TotalRounds:     100,
		CompletedRounds: 50,
		DetectionRate:   0.7,
		EvasionRate:     0.3,
		AttackStrategy:  "mixed",
		DefenderAlias:   "champion",
		CreatedAt:       "2026-01-18T12:00:00Z",
	}

	mockUC.On("GetByID", mock.Anything, battleID).Return(expectedOutput, nil)

	req, _ := http.NewRequest("GET", "/api/v1/battles/"+battleID.String(), nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	mockUC.AssertExpectations(t)
}

func TestGetBattle_NotFound(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	mockUC.On("GetByID", mock.Anything, battleID).Return(nil, usecase.ErrBattleNotFound)

	req, _ := http.NewRequest("GET", "/api/v1/battles/"+battleID.String(), nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.False(t, response.Success)
	assert.Equal(t, "NOT_FOUND", response.Error.Code)
}

func TestGetBattle_InvalidID(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	req, _ := http.NewRequest("GET", "/api/v1/battles/invalid-uuid", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.False(t, response.Success)
	assert.Equal(t, "INVALID_REQUEST", response.Error.Code)
}

func TestListBattles_Success(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	expectedOutput := &usecase.BattleListOutput{
		Battles: []*usecase.BattleOutput{
			{
				BattleID:       uuid.New(),
				Status:         "completed",
				TotalRounds:    100,
				AttackStrategy: "mixed",
			},
		},
		Total:   1,
		Limit:   20,
		Offset:  0,
		HasMore: false,
	}

	mockUC.On("List", mock.Anything, 20, 0).Return(expectedOutput, nil)

	req, _ := http.NewRequest("GET", "/api/v1/battles", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	mockUC.AssertExpectations(t)
}

func TestListBattles_WithPagination(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	expectedOutput := &usecase.BattleListOutput{
		Battles: []*usecase.BattleOutput{},
		Total:   50,
		Limit:   10,
		Offset:  20,
		HasMore: true,
	}

	mockUC.On("List", mock.Anything, 10, 20).Return(expectedOutput, nil)

	req, _ := http.NewRequest("GET", "/api/v1/battles?limit=10&offset=20", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	mockUC.AssertExpectations(t)
}

func TestStopBattle_Success(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	expectedOutput := &usecase.BattleOutput{
		BattleID:        battleID,
		Status:          "completed",
		TotalRounds:     100,
		CompletedRounds: 50,
		AttackStrategy:  "mixed",
		DefenderAlias:   "champion",
	}

	mockUC.On("Stop", mock.Anything, battleID).Return(expectedOutput, nil)

	req, _ := http.NewRequest("POST", "/api/v1/battles/"+battleID.String()+"/stop", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	mockUC.AssertExpectations(t)
}

func TestStopBattle_NotFound(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	mockUC.On("Stop", mock.Anything, battleID).Return(nil, usecase.ErrBattleNotFound)

	req, _ := http.NewRequest("POST", "/api/v1/battles/"+battleID.String()+"/stop", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestGetBattleStats_Success(t *testing.T) {
	mockUC := new(MockBattleUsecase)
	handler := NewBattleHandler(mockUC)
	router := setupTestRouter(handler)

	battleID := uuid.New()
	expectedOutput := &usecase.BattleOutput{
		BattleID:        battleID,
		Status:          "running",
		TotalRounds:     100,
		CompletedRounds: 75,
		DetectionRate:   0.72,
		EvasionRate:     0.28,
		AttackStrategy:  "leetspeak",
		DefenderAlias:   "champion",
	}

	mockUC.On("GetByID", mock.Anything, battleID).Return(expectedOutput, nil)

	req, _ := http.NewRequest("GET", "/api/v1/battles/"+battleID.String()+"/stats", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)

	data := response.Data.(map[string]interface{})
	assert.Equal(t, float64(100), data["total_rounds"])
	assert.Equal(t, float64(75), data["completed_rounds"])
	mockUC.AssertExpectations(t)
}
