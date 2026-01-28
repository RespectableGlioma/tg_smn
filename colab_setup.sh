#!/bin/bash
# Quick setup script for Google Colab
# Usage in Colab cell: !bash colab_setup.sh

set -e  # Exit on error

echo "=== TG-SMN Colab Setup ==="

# Configuration
REPO_URL="https://github.com/YOUR_USERNAME/tg_smn.git"
BRANCH="${BRANCH:-stoch-muzero-harness}"
REPO_DIR="/content/tg_smn"

# Clone or update repository
if [ -d "$REPO_DIR" ]; then
  echo "✓ Repository exists, updating..."
  cd "$REPO_DIR"
  git fetch --all
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  echo "✓ Cloning repository..."
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# Install package
echo "✓ Installing package..."
cd "$REPO_DIR"
pip install -q -e .

echo "✓ Setup complete!"
echo "Repository: $REPO_DIR"
echo "Branch: $BRANCH"
