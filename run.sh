#!/usr/bin/env bash
# Wrapper: loads secrets into THIS PROCESS's environment only (never printed,
# never written anywhere), then runs the bot.
#   ./run.sh                 -> dry-run against live tournaments (needs METACULUS_TOKEN for reads)
#   ./run.sh --submit        -> real forecasts (requires METACULUS_TOKEN)
# METACULUS_TOKEN can be provided via the environment or a local .env file
# (see .env.template). The OpenRouter key is read from ~/.openrouter/key.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/.openrouter/key" ]; then
  OPENROUTER_API_KEY="$(cat "$HOME/.openrouter/key")"
  export OPENROUTER_API_KEY
fi
exec python3 bot.py "$@"
