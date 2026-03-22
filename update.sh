#!/bin/bash
# Atualiza e reinicia a Lobeira.
# Uso: bash update.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "📥 Puxando atualizações..."
git pull origin main --ff-only

echo "📦 Atualizando dependências..."
.venv/bin/pip install -q -r requirements.txt

echo "🔁 Reiniciando serviços..."
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

sudo systemctl restart lobeira && echo "  ✅ lobeira"
sudo systemctl restart vl-ocr  && echo "  ✅ vl-ocr"

echo "✅ Pronto."
