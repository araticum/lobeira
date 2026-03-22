#!/bin/bash
# Setup inicial da Lobeira — rodar UMA vez após clonar.
# Instala: git hooks, serviços systemd, timer de auto-update.
#
# Uso: bash scripts/install-hooks.sh

set -e

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "🐍 Criando virtualenv..."
if [ ! -f "$REPO_ROOT/.venv/bin/python" ]; then
  python3 -m venv "$REPO_ROOT/.venv"
  echo "  ✅ .venv criado"
else
  echo "  ✓  .venv já existe"
fi

echo "📦 Instalando dependências..."
"$REPO_ROOT/.venv/bin/pip" install -q -r "$REPO_ROOT/requirements.txt"
echo "  ✅ dependências instaladas"

echo "⚙️  Configurando git hooks via core.hooksPath..."
git -C "$REPO_ROOT" config core.hooksPath scripts
chmod +x "$REPO_ROOT/scripts/post-merge"
echo "  ✅ Hooks ativos em scripts/ (sem cópia necessária)"

echo "📦 Instalando serviços systemd..."
sudo cp "$REPO_ROOT/lobeira.service" /etc/systemd/system/lobeira.service
sudo cp "$REPO_ROOT/vl-ocr.service"  /etc/systemd/system/vl-ocr.service
sudo systemctl daemon-reload

echo ""
echo "✅ Setup concluído. A partir de agora:"
echo "   • git pull → env vars + deps + restart automático via hook"
