#!/usr/bin/env bash
# Regenerates the sample runs in examples/ with a REAL model. Needs ANTHROPIC_API_KEY or OPENAI_API_KEY, git, network.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { dir="$1"; shift; python -m agent --out "examples/$dir" --no-cache "$@"; }

# 1. normal run
run run1_normal "Review the code quality of https://github.com/psf/requests and suggest fixes" || true
# 2. induced failures: timeouts, malformed response, 429 and a 404 on one tool call; the model must cope
run run2_injected_faults "Review https://github.com/pallets/flask and suggest fixes" \
  --inject-fault repo_list_files:timeout:2 --inject-fault repo_read_file:malformed:1 \
  --inject-fault repo_read_file:rate_limit:1 --inject-fault repo_read_file:not_found:1 || true
# 3. real failure: misspelled repository name that the model must notice and repair itself
run run3_misspelled_repo "Review github.com/psf/requets and suggest fixes" || true
# 4. ambiguous goal: should ask for clarification instead of guessing
run run4_ambiguous_goal "make my code better" || true
