#!/bin/bash
# EvoGuard E2E Battle Demo Script
# Demonstrates the full adversarial attack simulation flow

set -e

# Configuration
API_URL="${API_URL:-http://api.evoguard.local}"
ML_URL="${ML_URL:-http://ml.evoguard.local}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_demo() { echo -e "${CYAN}[DEMO]${NC} $1"; }
log_result() { echo -e "${YELLOW}[RESULT]${NC} $1"; }

# Check prerequisites
check_services() {
    log_step "Checking service availability..."

    # Check API service
    if curl -s "${API_URL}/health" > /dev/null 2>&1; then
        log_info "API Service: OK"
    else
        echo -e "${RED}API Service not available at ${API_URL}${NC}"
        echo "Run: minikube tunnel (in another terminal)"
        exit 1
    fi

    # Check ML service
    if curl -s "${ML_URL}/health" > /dev/null 2>&1; then
        log_info "ML Service: OK"
    else
        echo -e "${RED}ML Service not available at ${ML_URL}${NC}"
        exit 1
    fi
}

# Demo: Health checks
demo_health_checks() {
    echo ""
    echo "========================================="
    echo "  Step 1: Health Checks"
    echo "========================================="

    log_demo "Checking API service health..."
    curl -s "${API_URL}/health" | jq '.'

    log_demo "Checking ML service health..."
    curl -s "${ML_URL}/health" | jq '.'
}

# Demo: ML Classification
demo_classification() {
    echo ""
    echo "========================================="
    echo "  Step 2: Text Classification"
    echo "========================================="

    log_demo "Testing toxic content classification..."

    # Test benign text
    log_info "Testing benign text..."
    BENIGN_RESULT=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello, this is a friendly message about programming!"}')
    echo "$BENIGN_RESULT" | jq '.'

    # Test toxic text
    log_info "Testing potentially toxic text..."
    TOXIC_RESULT=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d '{"text": "You are absolutely terrible and should be ashamed of yourself!"}')
    echo "$TOXIC_RESULT" | jq '.'

    log_result "Classification shows the model can detect toxic content"
}

# Demo: Create Battle
demo_create_battle() {
    echo ""
    echo "========================================="
    echo "  Step 3: Create Adversarial Battle"
    echo "========================================="

    log_demo "Creating a new battle..."

    BATTLE_RESPONSE=$(curl -s -X POST "${API_URL}/api/v1/battles" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "E2E Demo Battle",
            "description": "Testing adversarial attack simulation",
            "attack_strategy": "textfooler",
            "max_rounds": 10,
            "target_model": "bert-toxic-classifier"
        }')

    echo "$BATTLE_RESPONSE" | jq '.'

    # Extract battle ID
    BATTLE_ID=$(echo "$BATTLE_RESPONSE" | jq -r '.id // .battle_id // empty')

    if [ -z "$BATTLE_ID" ]; then
        log_info "Note: Battle creation requires database. Showing mock flow..."
        BATTLE_ID="demo-battle-$(date +%s)"
    fi

    export BATTLE_ID
    log_result "Battle created with ID: $BATTLE_ID"
}

# Demo: Adversarial Attack Simulation
demo_attack_round() {
    echo ""
    echo "========================================="
    echo "  Step 4: Adversarial Attack Round"
    echo "========================================="

    log_demo "Simulating adversarial attack..."

    # Original text (classified as toxic)
    ORIGINAL_TEXT="You are stupid and terrible at everything"

    log_info "Original text: '$ORIGINAL_TEXT'"
    log_info "Running classification..."

    ORIGINAL_RESULT=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$ORIGINAL_TEXT\"}")

    echo "Original classification:"
    echo "$ORIGINAL_RESULT" | jq '.'

    # Simulated adversarial perturbation (character substitution)
    # In real scenario, attacker service would generate this
    ADVERSARIAL_TEXT="You are stup1d and terr1ble at everyth1ng"

    log_info "Adversarial text: '$ADVERSARIAL_TEXT'"
    log_info "Running classification on perturbed text..."

    ADVERSARIAL_RESULT=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$ADVERSARIAL_TEXT\"}")

    echo "Adversarial classification:"
    echo "$ADVERSARIAL_RESULT" | jq '.'

    # Compare results
    ORIGINAL_LABEL=$(echo "$ORIGINAL_RESULT" | jq -r '.label // .prediction // "unknown"')
    ADVERSARIAL_LABEL=$(echo "$ADVERSARIAL_RESULT" | jq -r '.label // .prediction // "unknown"')

    if [ "$ORIGINAL_LABEL" != "$ADVERSARIAL_LABEL" ]; then
        log_result "ATTACK SUCCESS: Model prediction changed from '$ORIGINAL_LABEL' to '$ADVERSARIAL_LABEL'"
    else
        log_result "ATTACK FAILED: Model maintained prediction '$ORIGINAL_LABEL' (robust!)"
    fi
}

# Demo: Feature Store
demo_feature_store() {
    echo ""
    echo "========================================="
    echo "  Step 5: Feature Store Integration"
    echo "========================================="

    log_demo "Testing feature store caching..."

    # First request (cache miss)
    log_info "First request (expected cache miss)..."
    START_TIME=$(date +%s%N)
    RESULT1=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message for feature caching"}')
    END_TIME=$(date +%s%N)
    DURATION1=$(( (END_TIME - START_TIME) / 1000000 ))
    log_info "Duration: ${DURATION1}ms"

    # Second request (cache hit)
    log_info "Second request (expected cache hit)..."
    START_TIME=$(date +%s%N)
    RESULT2=$(curl -s -X POST "${ML_URL}/classify" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message for feature caching"}')
    END_TIME=$(date +%s%N)
    DURATION2=$(( (END_TIME - START_TIME) / 1000000 ))
    log_info "Duration: ${DURATION2}ms"

    if [ "$DURATION2" -lt "$DURATION1" ]; then
        SPEEDUP=$(( DURATION1 / DURATION2 ))
        log_result "Feature caching provides ~${SPEEDUP}x speedup"
    else
        log_result "Caching may not be active or network variance"
    fi
}

# Demo: Monitoring Metrics
demo_monitoring() {
    echo ""
    echo "========================================="
    echo "  Step 6: Monitoring & Metrics"
    echo "========================================="

    log_demo "Checking model monitoring metrics..."

    # Get metrics endpoint
    METRICS=$(curl -s "${ML_URL}/metrics" 2>/dev/null || echo "Metrics endpoint not available")

    if echo "$METRICS" | grep -q "predictions_total"; then
        log_info "Prometheus metrics available:"
        echo "$METRICS" | grep -E "(predictions_total|prediction_latency|model_confidence)" | head -20
    else
        log_info "Metrics endpoint: $METRICS"
    fi

    log_result "Monitoring provides observability into model performance"
}

# Summary
print_summary() {
    echo ""
    echo "========================================="
    echo "  E2E Demo Complete!"
    echo "========================================="
    echo ""
    echo "What we demonstrated:"
    echo "  1. Health checks for all services"
    echo "  2. Text classification (toxic vs benign)"
    echo "  3. Battle creation for adversarial testing"
    echo "  4. Adversarial attack simulation"
    echo "  5. Feature store caching benefits"
    echo "  6. Monitoring and metrics collection"
    echo ""
    echo "Architecture:"
    echo "  ┌─────────────┐    ┌─────────────┐"
    echo "  │ api-service │───▶│ ml-service  │"
    echo "  │   (Go)      │    │  (Python)   │"
    echo "  └──────┬──────┘    └──────┬──────┘"
    echo "         │                  │"
    echo "  ┌──────▼──────┐    ┌──────▼──────┐"
    echo "  │ PostgreSQL  │    │    Redis    │"
    echo "  │ (battles)   │    │ (features)  │"
    echo "  └─────────────┘    └─────────────┘"
    echo ""
    echo "Next steps:"
    echo "  - View Grafana dashboards: kubectl port-forward svc/grafana 3000:3000"
    echo "  - Check logs: kubectl logs -f deploy/ml-service"
    echo "  - Scale up: kubectl scale deploy/ml-service --replicas=2"
    echo ""
}

# Main
main() {
    echo ""
    echo "╔═══════════════════════════════════════╗"
    echo "║     EvoGuard E2E Battle Demo          ║"
    echo "║  Adversarial Attack Simulation Flow   ║"
    echo "╚═══════════════════════════════════════╝"
    echo ""

    check_services
    demo_health_checks
    demo_classification
    demo_create_battle
    demo_attack_round
    demo_feature_store
    demo_monitoring
    print_summary
}

# Run main
main "$@"
