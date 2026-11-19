# Write-up: Repo-Review Agent

**Design.** A single model-driven loop. The model must first publish a plan (`make_plan`; only planning tools are
offered on turn 1), then chooses one tool at a time from a registry of typed, schema-described tools (git-backed
repo listing/reading/grep, an AST analyzer, a safe calculator) and sees the real result before deciding what to do
next. Nothing about the review is scripted: plan, tool order, interpretation of static-analysis hints, the score,
the findings and the fixes are all produced by the model and returned through a schema-validated `submit_report`
call (an invalid report is bounced back with the exact problems). Code owns only what a model should not: input
validation, timeouts, output-shape checks, automatic backoff retries for transient faults (timeout, 429,
malformed output), a turn budget, a plan-first gate, a warning when the same failing call repeats, forced
finalisation when the budget runs out, and tracing. Everything else that can go wrong (misspelled repo, wrong
path, empty result, bad arguments) is returned to the model as a typed error with a generic hint, and the model
fixes it, uses another tool, or calls `make_plan` again with a revised plan. The model API itself is retried with
backoff and degrades to an honest `failed` report rather than a crash. Two providers (Anthropic, any
OpenAI-compatible endpoint) sit behind one small interface; tools are executed with the repo checked out by shallow
`git clone` so there is no API quota and the analyzed code is never executed.

**Testing.** Offline tests drive the real loop and tools against a local git remote with scripted model decisions
(a clearly-labelled test double) to verify plan gating, retries, model-driven repair after a 404, loop guard,
report rejection/correction, clarification, budget exhaustion and API outage; adapter tests check both providers'
wire formats and error handling. They verify the machinery, not model quality.

**Limitations.** Output quality tracks the chosen model; there is no automatic evaluation of finding precision.
Analysis is Python-only and sampled by the model's own choices under a turn/size budget. Large repositories are
processed serially, and a hung tool call can only be abandoned (threads), not killed. I had no way to run the live
providers while building the submission, so the first real run is a smoke test and the sample runs must be
regenerated with `scripts/make_samples.sh`.

**With more time.** A small eval harness (synthetic repos with seeded defects -> precision/recall of reported
findings, plan-quality checks), a verification step where the model proposes a patch that is re-analysed, parallel
tool calls, token/cost budgeting, language-pluggable analyzers (ruff/semgrep in a sandbox), and streaming output.
