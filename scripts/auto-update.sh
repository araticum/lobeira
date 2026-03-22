#!/bin/bash
# Executado pelo lobeira-updater.service (systemd timer, a cada 5 min).
# Faz git pull — se houver mudanças, o hook post-merge cuida do restart.

set -e

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[auto-update] Verificando atualizações..."

# Busca sem aplicar
git fetch origin main --quiet

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
  echo "[auto-update] Sem novidades ($(git rev-parse --short HEAD))."
  exit 0
fi

echo "[auto-update] Nova versão detectada: $LOCAL → $REMOTE"
echo "[auto-update] Aplicando git pull..."
git pull origin main --ff-only

# O hook post-merge é disparado automaticamente pelo git pull acima.
echo "[auto-update] Pull concluído — hook post-merge executado."
