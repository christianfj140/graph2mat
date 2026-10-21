#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/agent-skills"

CLAUDE="$ROOT/.claude/skills"
CODEX="$ROOT/.agents/skills"

SKILLS=(
  siesta-matrix-physics
  graph2mat-validation
  hamiltonian-derivatives
  electron-phonon-validation
  materials-dataset-design
  scientific-experiment-audit
)

mkdir -p "$CLAUDE" "$CODEX"

for skill in "${SKILLS[@]}"; do
  src="$SOURCE/$skill"
  [[ -f "$src/SKILL.md" ]] || {
    echo "ERROR: missing $src/SKILL.md" >&2
    exit 1
  }

  rm -rf "$CLAUDE/$skill" "$CODEX/$skill"
  cp -a "$src" "$CLAUDE/$skill"
  cp -a "$src" "$CODEX/$skill"
done

echo "Synced ${#SKILLS[@]} managed skills to Claude Code and Codex project skill directories."
