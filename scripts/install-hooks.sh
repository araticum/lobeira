#!/bin/bash
# Instala os git hooks da Lobeira no repositório local.
# Rodar uma vez após clonar: bash scripts/install-hooks.sh

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

cp "$REPO_ROOT/scripts/post-merge" "$HOOKS_DIR/post-merge"
chmod +x "$HOOKS_DIR/post-merge"

echo "✅ Hook post-merge instalado em $HOOKS_DIR/post-merge"
