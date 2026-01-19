package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ressKim-io/EvoGuard/api-service/internal/usecase"
)

// BattleHandler handles battle-related HTTP requests
type BattleHandler struct {
	battleUC usecase.BattleUsecase
}

// NewBattleHandler creates a new battle handler
func NewBattleHandler(battleUC usecase.BattleUsecase) *BattleHandler {
	return &BattleHandler{battleUC: battleUC}
}

// CreateBattle handles POST /api/v1/battles
func (h *BattleHandler) CreateBattle(c *gin.Context) {
	var input usecase.CreateBattleInput
	if err := c.ShouldBindJSON(&input); err != nil {
		HandleInvalidRequest(c, err.Error())
		return
	}

	// Validate attack strategy
	if !IsValidAttackStrategy(input.AttackStrategy) {
		HandleInvalidRequest(c, "invalid attack_strategy")
		return
	}

	output, err := h.battleUC.Create(c.Request.Context(), &input)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusCreated, output)
}

// GetBattle handles GET /api/v1/battles/:id
func (h *BattleHandler) GetBattle(c *gin.Context) {
	id, err := ExtractUUIDParam(c, "id")
	if err != nil {
		HandleInvalidUUID(c, "battle id")
		return
	}

	output, err := h.battleUC.GetByID(c.Request.Context(), id)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// ListBattles handles GET /api/v1/battles
func (h *BattleHandler) ListBattles(c *gin.Context) {
	pagination := ParsePagination(c)

	output, err := h.battleUC.List(c.Request.Context(), pagination.Limit, pagination.Offset)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// StopBattle handles POST /api/v1/battles/:id/stop
func (h *BattleHandler) StopBattle(c *gin.Context) {
	id, err := ExtractUUIDParam(c, "id")
	if err != nil {
		HandleInvalidUUID(c, "battle id")
		return
	}

	output, err := h.battleUC.Stop(c.Request.Context(), id)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// GetBattleStats handles GET /api/v1/battles/:id/stats
func (h *BattleHandler) GetBattleStats(c *gin.Context) {
	id, err := ExtractUUIDParam(c, "id")
	if err != nil {
		HandleInvalidUUID(c, "battle id")
		return
	}

	output, err := h.battleUC.GetByID(c.Request.Context(), id)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	stats := map[string]interface{}{
		"battle_id":        output.BattleID,
		"status":           output.Status,
		"total_rounds":     output.TotalRounds,
		"completed_rounds": output.CompletedRounds,
		"detection_rate":   output.DetectionRate,
		"evasion_rate":     output.EvasionRate,
	}

	respondSuccess(c, http.StatusOK, stats)
}

// SubmitRound handles POST /api/v1/battles/:id/rounds
func (h *BattleHandler) SubmitRound(c *gin.Context) {
	id, err := ExtractUUIDParam(c, "id")
	if err != nil {
		HandleInvalidUUID(c, "battle id")
		return
	}

	var input usecase.SubmitRoundInput
	if err := c.ShouldBindJSON(&input); err != nil {
		HandleInvalidRequest(c, err.Error())
		return
	}

	output, err := h.battleUC.SubmitRound(c.Request.Context(), id, &input)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusCreated, output)
}

// GetRounds handles GET /api/v1/battles/:id/rounds
func (h *BattleHandler) GetRounds(c *gin.Context) {
	id, err := ExtractUUIDParam(c, "id")
	if err != nil {
		HandleInvalidUUID(c, "battle id")
		return
	}

	pagination := ParsePagination(c)

	rounds, total, err := h.battleUC.GetRounds(c.Request.Context(), id, pagination.Limit, pagination.Offset)
	if err != nil {
		HandleUsecaseError(c, err)
		return
	}

	respondSuccess(c, http.StatusOK, map[string]interface{}{
		"rounds":   rounds,
		"total":    total,
		"limit":    pagination.Limit,
		"offset":   pagination.Offset,
		"has_more": int64(pagination.Offset+pagination.Limit) < total,
	})
}
