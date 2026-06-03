# =============================================================================
# Cohere Portfolio — Dockerfile
# Packages the three Python demos as a single deployable container.
#
# Build:
#   docker build -t cohere-portfolio .
#
# Run all demos (mock mode, no API key needed):
#   docker run cohere-portfolio
#
# Run with real Cohere API key:
#   docker run -e COHERE_API_KEY=your_key_here cohere-portfolio
#
# Run a specific demo:
#   docker run -e COHERE_API_KEY=your_key cohere-portfolio python prompt-eval/prompt_eval.py --export
#   docker run -e COHERE_API_KEY=your_key cohere-portfolio python revops/revops_pipeline.py --sql
#   docker run -e COHERE_API_KEY=your_key cohere-portfolio python rag-agent/rag_agent.py --eval
#   docker run -e COHERE_API_KEY=your_key cohere-portfolio python prompt-eval/tool_agent.py --eval
#
# Interactive shell:
#   docker run -it cohere-portfolio /bin/bash
# =============================================================================

# Use slim Python 3.11 — smaller image, same compatibility
FROM python:3.11-slim

# Metadata
LABEL maintainer="Hai-Huong Le Vu"
LABEL description="Cohere Portfolio — Prompt Eval, RevOps Analytics, RAG Agent"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY prompt-eval/ ./prompt-eval/
COPY revops/      ./revops/
COPY rag-agent/   ./rag-agent/
COPY docs/        ./docs/
COPY README.md    .

# Create output directory for exported files
RUN mkdir -p /app/output

# Default command: run all three demos in sequence (mock mode if no API key)
CMD ["python", "prompt-eval/prompt_eval.py", "--mock"]
