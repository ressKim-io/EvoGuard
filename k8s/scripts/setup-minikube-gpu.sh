#!/bin/bash
# EvoGuard Minikube Setup Script with GPU Support
# This script sets up a local Kubernetes cluster with GPU capabilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELM_DIR="$PROJECT_ROOT/k8s/helm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check minikube
    if ! command -v minikube &> /dev/null; then
        log_error "minikube not found. Install: https://minikube.sigs.k8s.io/docs/start/"
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Install: https://helm.sh/docs/intro/install/"
        exit 1
    fi

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Install: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Install: https://docs.docker.com/engine/install/"
        exit 1
    fi

    # Check NVIDIA driver (for GPU)
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected"
        GPU_AVAILABLE=true
    else
        log_warn "NVIDIA GPU not detected. Will deploy in CPU-only mode."
        GPU_AVAILABLE=false
    fi

    log_info "All prerequisites satisfied"
}

# Start minikube with GPU support
start_minikube() {
    log_info "Starting minikube cluster..."

    # Check if minikube is already running
    if minikube status &> /dev/null; then
        log_info "Minikube is already running"
        return
    fi

    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Starting minikube with GPU support..."
        minikube start \
            --driver=docker \
            --gpus=all \
            --memory=8192 \
            --cpus=4 \
            --disk-size=30g \
            --kubernetes-version=v1.28.0
    else
        log_info "Starting minikube without GPU..."
        minikube start \
            --driver=docker \
            --memory=6144 \
            --cpus=4 \
            --disk-size=20g \
            --kubernetes-version=v1.28.0
    fi

    log_info "Minikube started successfully"
}

# Install NVIDIA device plugin for GPU support
install_nvidia_plugin() {
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Installing NVIDIA device plugin..."
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

        # Wait for plugin to be ready
        log_info "Waiting for NVIDIA device plugin to be ready..."
        kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=120s || true
    fi
}

# Enable required addons
enable_addons() {
    log_info "Enabling minikube addons..."

    minikube addons enable ingress
    minikube addons enable metrics-server
    minikube addons enable storage-provisioner

    log_info "Addons enabled"
}

# Add Helm repositories
add_helm_repos() {
    log_info "Adding Helm repositories..."

    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update

    log_info "Helm repositories updated"
}

# Build local Docker images
build_images() {
    log_info "Building local Docker images..."

    # Point to minikube's Docker daemon
    eval $(minikube docker-env)

    # Build api-service
    log_info "Building api-service..."
    docker build -t evoguard/api-service:local "$PROJECT_ROOT/api-service"

    # Build ml-service
    log_info "Building ml-service..."
    docker build -t evoguard/ml-service:local "$PROJECT_ROOT/ml-service"

    log_info "Docker images built"
}

# Deploy shared services
deploy_shared_services() {
    log_info "Deploying shared services (PostgreSQL, Redis)..."

    cd "$HELM_DIR/shared-services"

    # Update dependencies
    helm dependency update

    # Install/upgrade
    helm upgrade --install shared-services . \
        -f values-local.yaml \
        --wait \
        --timeout 5m

    log_info "Shared services deployed"
}

# Deploy api-service
deploy_api_service() {
    log_info "Deploying api-service..."

    cd "$HELM_DIR/api-service"

    helm upgrade --install api-service . \
        -f values-local.yaml \
        --wait \
        --timeout 3m

    log_info "api-service deployed"
}

# Deploy ml-service
deploy_ml_service() {
    log_info "Deploying ml-service..."

    cd "$HELM_DIR/ml-service"

    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Deploying with GPU support..."
        helm upgrade --install ml-service . \
            -f values-local.yaml \
            --wait \
            --timeout 10m
    else
        log_info "Deploying CPU-only version..."
        helm upgrade --install ml-service . \
            -f values-local-cpu.yaml \
            --wait \
            --timeout 10m
    fi

    log_info "ml-service deployed"
}

# Setup /etc/hosts
setup_hosts() {
    log_info "Setting up local DNS..."

    MINIKUBE_IP=$(minikube ip)

    echo ""
    echo "Add the following lines to /etc/hosts (requires sudo):"
    echo ""
    echo "$MINIKUBE_IP api.evoguard.local"
    echo "$MINIKUBE_IP ml.evoguard.local"
    echo ""

    read -p "Add entries automatically? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove old entries
        sudo sed -i '/evoguard.local/d' /etc/hosts
        # Add new entries
        echo "$MINIKUBE_IP api.evoguard.local" | sudo tee -a /etc/hosts
        echo "$MINIKUBE_IP ml.evoguard.local" | sudo tee -a /etc/hosts
        log_info "Hosts file updated"
    fi
}

# Print status
print_status() {
    echo ""
    echo "========================================="
    echo "  EvoGuard K8s Deployment Complete!"
    echo "========================================="
    echo ""
    echo "Services:"
    kubectl get pods
    echo ""
    echo "Endpoints:"
    echo "  API Service:  http://api.evoguard.local"
    echo "  ML Service:   http://ml.evoguard.local"
    echo ""
    echo "Useful commands:"
    echo "  kubectl get pods              # List pods"
    echo "  kubectl logs -f <pod-name>    # View logs"
    echo "  minikube dashboard            # Open dashboard"
    echo "  minikube tunnel               # Expose LoadBalancer services"
    echo ""
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "GPU Status:"
        kubectl describe nodes | grep -A5 "nvidia.com/gpu"
    fi
    echo ""
}

# Main
main() {
    echo ""
    echo "========================================="
    echo "  EvoGuard Minikube Setup"
    echo "========================================="
    echo ""

    check_prerequisites
    start_minikube
    install_nvidia_plugin
    enable_addons
    add_helm_repos
    build_images
    deploy_shared_services
    deploy_api_service
    deploy_ml_service
    setup_hosts
    print_status
}

# Run main
main "$@"
