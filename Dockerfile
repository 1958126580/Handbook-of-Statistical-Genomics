# ============================================================================
# StatisticalGenomics.jl - Production Docker Image
# ============================================================================
# Multi-stage build for optimized production deployment
# Based on official Julia image with comprehensive genomics tooling
# ============================================================================

# Stage 1: Builder - Compile and precompile all dependencies
FROM julia:1.10-bullseye AS builder

# Set environment variables
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV JULIA_NUM_THREADS=auto
ENV JULIA_PROJECT=/app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy project files first for better caching
COPY Project.toml .
COPY src/ src/

# Instantiate and precompile all dependencies
RUN julia --project=. -e '\
    using Pkg; \
    Pkg.instantiate(); \
    Pkg.precompile(); \
    # Create system image for faster startup \
    using StatisticalGenomics; \
    '

# Stage 2: Production - Minimal runtime image
FROM julia:1.10-bullseye AS production

# Labels for container metadata
LABEL maintainer="StatisticalGenomics Team"
LABEL version="1.0.0"
LABEL description="Comprehensive statistical genomics analysis platform"
LABEL org.opencontainers.image.source="https://github.com/statgen/StatisticalGenomics.jl"

# Set environment variables
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV JULIA_NUM_THREADS=auto
ENV JULIA_PROJECT=/app
ENV STATGEN_DATA_DIR=/data
ENV STATGEN_OUTPUT_DIR=/output
ENV STATGEN_CONFIG_DIR=/config

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    liblapack3 \
    libhdf5-103 \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libcurl4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r statgen && useradd -r -g statgen statgen

# Create necessary directories
RUN mkdir -p /app /data /output /config /logs \
    && chown -R statgen:statgen /app /data /output /config /logs

# Copy precompiled Julia depot from builder
COPY --from=builder /opt/julia-depot /opt/julia-depot

# Copy application code
WORKDIR /app
COPY --chown=statgen:statgen . .

# Switch to non-root user
USER statgen

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD julia --project=. -e 'using StatisticalGenomics; println("OK")' || exit 1

# Default entry point
ENTRYPOINT ["julia", "--project=."]

# Default command shows help
CMD ["-e", "using StatisticalGenomics; println(StatisticalGenomics.help_text())"]

# ============================================================================
# Build instructions:
# docker build -t statgen/statisticalgenomics:latest .
#
# Run examples:
# docker run -v /path/to/data:/data statgen/statisticalgenomics:latest script.jl
# docker run -it statgen/statisticalgenomics:latest  # Interactive Julia REPL
# ============================================================================
