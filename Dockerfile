# -----------------------------
# Base Python image
# -----------------------------
FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Expose FastAPI port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Start FastAPI server
# -----------------------------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
