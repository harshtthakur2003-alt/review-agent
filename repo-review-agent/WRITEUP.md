# Write-up: Repo-Review Agent

**Design decisions.** The agent is a plan-then-execute loop with a separate planner, executor, tool registry and
reporter, so each part can be tested and swapped independently. The planner returns a *structured* plan (steps,
tools, `$ref` data-flow, `foreach` fan-out) that is validated before anything runs (known tools, no forward
references) and printed as the planning trace. Invalid LLM plans are fed back once, then replaced by a template.
Tools are thin and typed: the registry checks required inputs, enforces a hard timeout and validates the output
shape, and every failure is classified (`transient`, `rate_limited`, `not_found`, `invalid_*`, `fatal`). That
taxonomy drives a bounded **recovery ladder**: backoff retries for transient/malformed results, deterministic
tool fallbacks for `not_found` (misspelled repo → fuzzy match against the owner's repos, accepted only if
unambiguous; wrong branch → `main`/`master`), then an optional LLM repair proposal, then graceful degradation:
dependent steps are skipped, per-file failures yield a `partial` result, and everything lands in the report's
execution log. Autonomous repairs are surfaced as *assumptions* rather than hidden. Repo contents come from
shallow `git clone` (no API quota, easy to test against local repos); analysis is pure AST so untrusted code is
never executed, and the calculator is an AST whitelist (no `eval`). A fault injector and 40 offline tests
(including end-to-end runs against a real local git remote) make the recovery behaviour verifiable.

**Limitations.** (1) The offline backend is a fixed template, not a reasoning model, and the real
`AnthropicLLM` path was only tested against a stub, since no API key was available while building. So the
"autonomous decomposition" claim is only fully exercised with a live model. (2) Analysis is heuristic and
sampled: the 8 largest non-test files, AST-only checks, approximate cyclomatic complexity, and a score with
arbitrary severity weights. It finds smells, not bugs, and is Python-only. (3) Plans are linear; there is no
mid-run *re-planning* when a step fails permanently (the agent degrades instead). (4) Timeouts use threads,
which cannot kill a stuck call, only stop waiting for it. (5) No concurrency; large repos are processed serially.

**With more time.** Run the live-LLM path and add a small eval harness (known-bad synthetic repos with seeded
defects → precision/recall of findings; plan-quality checks across varied goals). Add true re-planning (give
the LLM the failed step plus state and let it propose a replacement sub-plan, still validated). Let the LLM
triage and explain top findings with code context, and propose actual patches that are checked by re-running the
analyzer. Add language-pluggable analyzers (ruff/semgrep as sandboxed subprocess tools), parallel file
fetching, token/cost budgeting, and structured-output (tool-use) mode for the planner instead of JSON-in-text.
