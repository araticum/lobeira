#!/bin/bash
# Setup inicial da Lobeira — rodar UMA vez após clonar.
# Instala: git hooks, serviços systemd, timer de auto-update.
#
# Uso: bash scripts/install-hooks.sh

set -e

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "⚙️  Configurando git hooks via core.hooksPath..."
git -C "$REPO_ROOT" config core.hooksPath scripts
chmod +x "$REPO_ROOT/scripts/post-merge"
echo "  ✅ Hooks ativos em scripts/ (sem cópia necessária)"

echo "📦 Instalando serviços systemd..."
sudo cp "$REPO_ROOT/lobeira.service"         /etc/systemd/system/lobeira.service
sudo cp "$REPO_ROOT/vl-ocr.service"          /etc/systemd/system/vl-ocr.service
sudo cp "$REPO_ROOT/scripts/lobeira-updater.service" /etc/systemd/system/lobeira-updater.service
sudo cp "$REPO_ROOT/scripts/lobeira-updater.timer"   /etc/systemd/system/lobeira-updater.timer
sudo systemctl daemon-reload

echo "🔁 Habilitando auto-update (timer a cada 5 min)..."
sudo systemctl enable --now lobeira-updater.timer
echo "  ✅ Timer ativo"

echo ""
echo "✅ Setup concluído. A partir de agora:"
echo "   • git pull → reinicia automaticamente via hook"
echo "   • timer    → puxa atualizações a cada 5 minutos sozinho"
