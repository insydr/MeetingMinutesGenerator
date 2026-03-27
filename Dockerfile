# =============================================================================
# Dockerfile for Meeting Minutes Generator
# Optimized for Hugging Face Spaces deployment
# =============================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create samples directory if it doesn't exist
RUN mkdir -p samples

# Set environment variables for CPU optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MEETING_MINUTES_MOCK_MODE=false
ENV MEETING_MINUTES_DEBUG=false

# Expose port for Gradio
EXPOSE 7860

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]
