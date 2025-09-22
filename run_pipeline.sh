#!/bin/bash
# ===========================
# 🚀 Universal Pipeline Runner
# ===========================

set -e  # stop if any command fails

# -------- CONFIG --------
ENV_NAME="finbert_env"
SCRIPT="main.py"

echo "🔹 Starting pipeline setup..."

# -------- ENV DETECTION --------
if [ -n "$STUDIO_NAME" ] || [ -n "$HF_HOME" ]; then
    echo "⚠️ Running inside a managed Studio (venv/conda not allowed)."
    echo "🔹 Using the default environment..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "🔹 Running on a normal server → creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# -------- HUGGINGFACE AUTH --------
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ No HF_TOKEN found. Please set it with: export HF_TOKEN=your_token"
else
    echo "🔹 Hugging Face token detected."
    export HF_TOKEN=$HF_TOKEN
fi

# -------- RUN SCRIPT --------
echo "🚀 Launching pipeline..."
python $SCRIPT "$@"

echo "✅ Pipeline finished successfully!"
