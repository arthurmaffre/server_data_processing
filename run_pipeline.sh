#!/bin/bash
# ===========================
# 🚀 Batch script for server
# ===========================

set -e  # stop if any command fails

# -------- CONFIG --------
ENV_NAME="finbert_env"
SCRIPT="main.py"

# If not already set in shell, you can set your HF token here
export HF_TOKEN="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"

# -------- ENV SETUP --------
if [ ! -d "$ENV_NAME" ]; then
    echo "🔹 Creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
fi

echo "🔹 Activating virtual environment..."
source $ENV_NAME/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing dependencies..."
pip install -r requirements.txt

# -------- HUGGINGFACE AUTH --------
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN is not set. Please export your Hugging Face token."
    echo "   Example: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx"
    exit 1
else
    echo "🔹 Hugging Face token detected."
fi

# -------- RUN SCRIPT --------
echo "🚀 Launching pipeline..."
python $SCRIPT "$@"

echo "✅ Pipeline finished successfully!"
