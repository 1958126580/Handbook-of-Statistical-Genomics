#!/bin/bash
# ============================================================================
# StatisticalGenomics.jl - Deployment Script
# ============================================================================
# Comprehensive deployment automation for various environments
# Usage: ./deploy.sh [environment] [options]
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${STATGEN_VERSION:-1.0.0}"
REGISTRY="${STATGEN_REGISTRY:-statgen}"
IMAGE_NAME="statisticalgenomics"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Display usage information
usage() {
    cat << EOF
StatisticalGenomics.jl Deployment Script

Usage: $(basename "$0") <command> [options]

Commands:
    local       Deploy locally using Docker Compose
    build       Build Docker images
    push        Push images to registry
    kubernetes  Deploy to Kubernetes cluster
    hpc         Generate HPC job scripts
    test        Run deployment tests
    clean       Clean up deployment artifacts

Options:
    -e, --env ENV       Environment (dev, staging, prod)
    -v, --version VER   Version tag (default: $VERSION)
    -r, --registry REG  Container registry (default: $REGISTRY)
    -n, --namespace NS  Kubernetes namespace
    -h, --help          Show this help message

Examples:
    $(basename "$0") local -e dev
    $(basename "$0") build -v 1.0.0
    $(basename "$0") kubernetes -e prod -n statgen
    $(basename "$0") hpc --nodes 10 --time 24:00:00
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing+=("docker-compose")
    command -v julia >/dev/null 2>&1 || missing+=("julia")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Build Docker images
build_images() {
    local env="${1:-prod}"
    log_info "Building Docker images for environment: $env"

    cd "$PROJECT_ROOT"

    # Build main image
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --tag "$REGISTRY/$IMAGE_NAME:$VERSION" \
        --tag "$REGISTRY/$IMAGE_NAME:latest" \
        .

    # Build notebook image if Dockerfile exists
    if [[ -f Dockerfile.notebook ]]; then
        docker build \
            --file Dockerfile.notebook \
            --tag "$REGISTRY/$IMAGE_NAME-notebook:$VERSION" \
            --tag "$REGISTRY/$IMAGE_NAME-notebook:latest" \
            .
    fi

    log_success "Images built successfully"
}

# Deploy locally
deploy_local() {
    local env="${1:-dev}"
    log_info "Deploying locally with environment: $env"

    cd "$PROJECT_ROOT"

    # Create necessary directories
    mkdir -p data output logs config secrets

    # Generate secrets if they don't exist
    if [[ ! -f secrets/db_password.txt ]]; then
        openssl rand -base64 32 > secrets/db_password.txt
        log_info "Generated database password"
    fi

    if [[ ! -f secrets/grafana_password.txt ]]; then
        openssl rand -base64 32 > secrets/grafana_password.txt
        log_info "Generated Grafana password"
    fi

    # Start services
    docker-compose up -d

    log_success "Local deployment started"
    log_info "Services:"
    log_info "  - Jupyter Notebook: http://localhost:8888"
    log_info "  - Grafana: http://localhost:3000"
    log_info "  - Prometheus: http://localhost:9090"
}

# Generate Kubernetes manifests
deploy_kubernetes() {
    local env="${1:-prod}"
    local namespace="${2:-statgen}"

    log_info "Generating Kubernetes manifests for environment: $env"

    mkdir -p "$PROJECT_ROOT/k8s/$env"

    # Generate namespace
    cat > "$PROJECT_ROOT/k8s/$env/namespace.yaml" << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $namespace
  labels:
    app: statgen
    environment: $env
EOF

    # Generate deployment
    cat > "$PROJECT_ROOT/k8s/$env/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: statgen-worker
  namespace: $namespace
  labels:
    app: statgen
    component: worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: statgen
      component: worker
  template:
    metadata:
      labels:
        app: statgen
        component: worker
    spec:
      containers:
      - name: statgen
        image: $REGISTRY/$IMAGE_NAME:$VERSION
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "64Gi"
            cpu: "16"
        volumeMounts:
        - name: data
          mountPath: /data
          readOnly: true
        - name: output
          mountPath: /output
        env:
        - name: JULIA_NUM_THREADS
          value: "auto"
        - name: STATGEN_LOG_LEVEL
          value: "INFO"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: statgen-data
      - name: output
        persistentVolumeClaim:
          claimName: statgen-output
---
apiVersion: v1
kind: Service
metadata:
  name: statgen-service
  namespace: $namespace
spec:
  selector:
    app: statgen
    component: worker
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
EOF

    # Generate PVC
    cat > "$PROJECT_ROOT/k8s/$env/pvc.yaml" << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: statgen-data
  namespace: $namespace
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Ti
  storageClassName: fast-storage
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: statgen-output
  namespace: $namespace
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-storage
EOF

    log_success "Kubernetes manifests generated in k8s/$env/"
    log_info "Apply with: kubectl apply -f k8s/$env/"
}

# Generate HPC job scripts
generate_hpc_scripts() {
    local nodes="${1:-1}"
    local time="${2:-24:00:00}"
    local partition="${3:-normal}"

    log_info "Generating HPC job scripts"

    mkdir -p "$PROJECT_ROOT/hpc"

    # Slurm script
    cat > "$PROJECT_ROOT/hpc/slurm_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=statgen_analysis
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=$time
#SBATCH --partition=$partition
#SBATCH --output=logs/statgen_%j.out
#SBATCH --error=logs/statgen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=\$USER@institution.edu

# Load modules
module load julia/1.10
module load openblas/0.3.21

# Set environment
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export JULIA_DEPOT_PATH=\$SCRATCH/.julia

# Change to project directory
cd \$SLURM_SUBMIT_DIR

# Run analysis
julia --project=. scripts/run_analysis.jl

echo "Job completed at \$(date)"
EOF

    # PBS script
    cat > "$PROJECT_ROOT/hpc/pbs_job.sh" << EOF
#!/bin/bash
#PBS -N statgen_analysis
#PBS -l nodes=$nodes:ppn=32
#PBS -l mem=128gb
#PBS -l walltime=$time
#PBS -q $partition
#PBS -o logs/statgen_\${PBS_JOBID}.out
#PBS -e logs/statgen_\${PBS_JOBID}.err
#PBS -m abe
#PBS -M \$USER@institution.edu

# Load modules
module load julia/1.10
module load openblas

# Set environment
export JULIA_NUM_THREADS=32
export JULIA_DEPOT_PATH=\$SCRATCH/.julia

# Change to project directory
cd \$PBS_O_WORKDIR

# Run analysis
julia --project=. scripts/run_analysis.jl

echo "Job completed at \$(date)"
EOF

    # LSF script
    cat > "$PROJECT_ROOT/hpc/lsf_job.sh" << EOF
#!/bin/bash
#BSUB -J statgen_analysis
#BSUB -n 32
#BSUB -R "span[ptile=32]"
#BSUB -M 128000
#BSUB -W $time
#BSUB -q $partition
#BSUB -o logs/statgen_%J.out
#BSUB -e logs/statgen_%J.err

# Load modules
module load julia/1.10
module load openblas

# Set environment
export JULIA_NUM_THREADS=32

# Change to project directory
cd \$LS_SUBCWD

# Run analysis
julia --project=. scripts/run_analysis.jl

echo "Job completed at \$(date)"
EOF

    chmod +x "$PROJECT_ROOT/hpc/"*.sh
    log_success "HPC scripts generated in hpc/"
}

# Run deployment tests
run_tests() {
    log_info "Running deployment tests..."

    cd "$PROJECT_ROOT"

    # Test Julia package
    julia --project=. -e '
        using Pkg
        Pkg.test()
    '

    # Test Docker build
    docker build --target production -t statgen-test .
    docker run --rm statgen-test -e 'using StatisticalGenomics; println("Docker test passed")'
    docker rmi statgen-test

    log_success "All deployment tests passed"
}

# Clean up
clean() {
    log_info "Cleaning up deployment artifacts..."

    cd "$PROJECT_ROOT"

    # Stop and remove containers
    docker-compose down -v --remove-orphans 2>/dev/null || true

    # Remove built images
    docker rmi "$REGISTRY/$IMAGE_NAME:$VERSION" 2>/dev/null || true
    docker rmi "$REGISTRY/$IMAGE_NAME:latest" 2>/dev/null || true

    # Clean Julia packages
    rm -rf Manifest.toml

    log_success "Cleanup complete"
}

# Main entry point
main() {
    local command="${1:-}"
    shift || true

    local env="prod"
    local namespace="statgen"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--env)
                env="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -n|--namespace)
                namespace="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done

    case "$command" in
        local)
            check_prerequisites
            deploy_local "$env"
            ;;
        build)
            check_prerequisites
            build_images "$env"
            ;;
        push)
            docker push "$REGISTRY/$IMAGE_NAME:$VERSION"
            docker push "$REGISTRY/$IMAGE_NAME:latest"
            log_success "Images pushed to registry"
            ;;
        kubernetes)
            deploy_kubernetes "$env" "$namespace"
            ;;
        hpc)
            generate_hpc_scripts
            ;;
        test)
            check_prerequisites
            run_tests
            ;;
        clean)
            clean
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
