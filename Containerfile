# Multi-stage build for Managed Notifications Search MCP Server
# Stage 1: Build embeddings (only rebuilds when notifications or script changes)
FROM python:3.13-slim as embeddings-builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy all source files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies and project
RUN uv sync --frozen --no-dev

# Pre-download the sentence transformer model
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build embeddings database
COPY scripts/ ./scripts/
COPY managed-notifications/ ./managed-notifications/
RUN uv run scripts/build_embeddings.py

# Stage 2: Runtime image with MCP server
FROM python:3.13-slim as runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy all source files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies and project
RUN uv sync --frozen --no-dev

# Pre-download the sentence transformer model
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy pre-built embeddings database from builder stage
COPY --from=embeddings-builder /app/chroma_db/ ./chroma_db/

# Set environment variables for MCP server
ENV HOST=0.0.0.0
ENV PORT=8000

# Copy mcp server code
COPY main.py ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mcp
RUN chown -R mcp:mcp /app
USER mcp

# Expose port for MCP server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import chromadb; client = chromadb.PersistentClient('/app/chroma_db'); client.get_collection('managed_notifications')" || exit 1

# Default command
CMD ["uv", "run", "main.py"]
