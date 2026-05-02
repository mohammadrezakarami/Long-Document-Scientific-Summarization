#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

curl -sS \
  -X POST "${BASE_URL}/summarize" \
  -H "Content-Type: application/json" \
  --data @"${SCRIPT_DIR}/api_request.json"
