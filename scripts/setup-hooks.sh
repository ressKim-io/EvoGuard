#!/bin/bash
# Setup git hooks for the project
# Run this after cloning the repository

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Setting up git hooks..."

# Copy pre-commit hook
cp "$PROJECT_ROOT/scripts/git-hooks/pre-commit" "$PROJECT_ROOT/.git/hooks/pre-commit"
chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

echo "Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  - pre-commit: runs lint and test before commit"
