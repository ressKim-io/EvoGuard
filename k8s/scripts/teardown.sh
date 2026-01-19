#!/bin/bash
# Teardown EvoGuard K8s deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

log_info "Uninstalling Helm releases..."

helm uninstall ml-service 2>/dev/null || true
helm uninstall api-service 2>/dev/null || true
helm uninstall shared-services 2>/dev/null || true

log_info "Deleting PVCs..."
kubectl delete pvc --all 2>/dev/null || true

log_info "Cleaning up secrets..."
kubectl delete secret api-service-db-secret ml-service-registry-secret 2>/dev/null || true

log_info "Done! Cluster is clean."
echo ""
echo "To completely remove minikube: minikube delete"
