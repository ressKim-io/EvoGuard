package handler

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

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
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate attack strategy
	validStrategies := map[string]bool{
		"unicode_evasion": true,
		"homoglyph":       true,
		"leetspeak":       true,
		"llm_evasion":     true,
		"adversarial_llm": true,
		"mixed":           true,
	}
	if !validStrategies[input.AttackStrategy] {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid attack_strategy")
		return
	}

	output, err := h.battleUC.Create(c.Request.Context(), &input)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}

	respondSuccess(c, http.StatusCreated, output)
}

// GetBattle handles GET /api/v1/battles/:id
func (h *BattleHandler) GetBattle(c *gin.Context) {
	idStr := c.Param("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid battle id")
		return
	}

	output, err := h.battleUC.GetByID(c.Request.Context(), id)
	if err != nil {
		if err == usecase.ErrBattleNotFound {
			respondError(c, http.StatusNotFound, "NOT_FOUND", "battle not found")
			return
		}
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// ListBattles handles GET /api/v1/battles
func (h *BattleHandler) ListBattles(c *gin.Context) {
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "20"))
	if err != nil {
		limit = 20
	}
	offset, err := strconv.Atoi(c.DefaultQuery("offset", "0"))
	if err != nil {
		offset = 0
	}

	output, err := h.battleUC.List(c.Request.Context(), limit, offset)
	if err != nil {
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// StopBattle handles POST /api/v1/battles/:id/stop
func (h *BattleHandler) StopBattle(c *gin.Context) {
	idStr := c.Param("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid battle id")
		return
	}

	output, err := h.battleUC.Stop(c.Request.Context(), id)
	if err != nil {
		if err == usecase.ErrBattleNotFound {
			respondError(c, http.StatusNotFound, "NOT_FOUND", "battle not found")
			return
		}
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}

	respondSuccess(c, http.StatusOK, output)
}

// GetBattleStats handles GET /api/v1/battles/:id/stats
func (h *BattleHandler) GetBattleStats(c *gin.Context) {
	idStr := c.Param("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid battle id")
		return
	}

	output, err := h.battleUC.GetByID(c.Request.Context(), id)
	if err != nil {
		if err == usecase.ErrBattleNotFound {
			respondError(c, http.StatusNotFound, "NOT_FOUND", "battle not found")
			return
		}
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
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
	idStr := c.Param("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid battle id")
		return
	}

	var input usecase.SubmitRoundInput
	if err := c.ShouldBindJSON(&input); err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	output, err := h.battleUC.SubmitRound(c.Request.Context(), id, &input)
	if err != nil {
		switch err {
		case usecase.ErrBattleNotFound:
			respondError(c, http.StatusNotFound, "NOT_FOUND", "battle not found")
		case usecase.ErrBattleNotRunnable:
			respondError(c, http.StatusConflict, "CONFLICT", "battle cannot accept rounds")
		case usecase.ErrBattleCompleted:
			respondError(c, http.StatusConflict, "CONFLICT", "battle already completed")
		default:
			respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		}
		return
	}

	respondSuccess(c, http.StatusCreated, output)
}

// GetRounds handles GET /api/v1/battles/:id/rounds
func (h *BattleHandler) GetRounds(c *gin.Context) {
	idStr := c.Param("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		respondError(c, http.StatusBadRequest, "INVALID_REQUEST", "invalid battle id")
		return
	}

	limit, err := strconv.Atoi(c.DefaultQuery("limit", "20"))
	if err != nil {
		limit = 20
	}
	offset, err := strconv.Atoi(c.DefaultQuery("offset", "0"))
	if err != nil {
		offset = 0
	}

	rounds, total, err := h.battleUC.GetRounds(c.Request.Context(), id, limit, offset)
	if err != nil {
		if err == usecase.ErrBattleNotFound {
			respondError(c, http.StatusNotFound, "NOT_FOUND", "battle not found")
			return
		}
		respondError(c, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}

	respondSuccess(c, http.StatusOK, map[string]interface{}{
		"rounds":   rounds,
		"total":    total,
		"limit":    limit,
		"offset":   offset,
		"has_more": int64(offset+limit) < total,
	})
}
