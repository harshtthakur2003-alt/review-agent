# Code review report

**Goal:** Review the code quality of https://github.com/pallets/flask  
**Status:** `partial`  
**Planner/LLM backend:** offline-heuristic

**Repository:** pallets/flask (branch `main`)

## Result
- **Quality score:** 82.3 / 100
- Files analysed: 7 (3386 LOC)
- Issues: 38 (high 2, medium 18, low 18)

## Issues by type
| Issue | Count | Suggested fix |
|---|---|---|
| missing_docstring | 18 | Add short docstrings to public modules/classes/functions. |
| long_function | 7 | Extract cohesive blocks into helpers; aim for functions that fit on one screen. |
| high_complexity | 5 | Split into smaller functions / early returns to reduce branching. |
| too_many_args | 3 | Group related parameters into a dataclass or config object. |
| swallowed_exception | 3 | Log the exception or re-raise; silent `pass` hides real bugs. |
| eval_exec | 2 | Replace eval/exec with ast.literal_eval or an explicit dispatch table. |

## Top issues
| Severity | Location | Detail |
|---|---|---|
| high | `src/flask/cli.py:1022` | call to eval() |
| high | `src/flask/config.py:209` | call to exec() |
| medium | `src/flask/cli.py:41` | find_best_app() cyclomatic complexity ~12 (limit 10) |
| medium | `src/flask/cli.py:120` | find_app_by_string() is 78 lines (limit 60) |
| medium | `src/flask/cli.py:120` | find_app_by_string() cyclomatic complexity ~14 (limit 10) |
| medium | `src/flask/cli.py:698` | load_dotenv() is 66 lines (limit 60) |
| medium | `src/flask/cli.py:698` | load_dotenv() cyclomatic complexity ~13 (limit 10) |
| medium | `src/flask/cli.py:934` | run_command() takes 9 parameters |
| medium | `src/flask/cli.py:1035` | exception caught and silently ignored (pass) |
| medium | `src/flask/cli.py:1060` | routes_command() cyclomatic complexity ~18 (limit 10) |

## Hotspot files
- `src/flask/cli.py`: 18 issues in 825 LOC
- `src/flask/helpers.py`: 5 issues in 510 LOC
- `src/flask/sansio/blueprints.py`: 4 issues in 557 LOC

## What I did and why
The goal is a code-quality review of a GitHub repository. I must (1) resolve the repo and its default branch, (2) enumerate Python files, (3) pick a bounded, representative sample of source files, (4) fetch and statically analyse each, (5) aggregate, (6) compute a score, then report.

1. Resolve repository and default branch (`repo_info`)
2. List Python files on the default branch (`repo_list_files`)
3. Select a representative sample of source files (`select_python_files`)
4. Fetch each selected file (`repo_read_file`)
5. Statically analyse each file (`analyze_code`)
6. Aggregate findings (`summarize_findings`)
7. Compute weighted issues per 1,000 LOC (`calculator`)
8. Derive 0-100 quality score (`calculator`)

## Execution log
| Step | Status | Attempts | Recoveries / notes |
|---|---|---|---|
| s1 | ok | 1 | - |
| s2 | ok | 3 | transient error -> retry in 0.5s (exponential backoff); transient error -> retry in 1.0s (exponential backoff) |
| s3 | ok | 1 | - |
| s4 | partial | 10 | invalid_output error -> retry in 0.5s (exponential backoff); rate_limited error -> retry in 1.0s (exponential backoff); 1/8 items failed and were excluded |
| s5 | ok | 7 | - |
| s6 | ok | 1 | - |
| s7 | ok | 1 | - |
| s8 | ok | 1 | - |

## Assumptions
- Only Python files are reviewed.
- A sample of the largest non-test source files is analysed (bounded cost) rather than the whole repository.
- Analysis is static (AST); code is never executed.

## Limitations
- Static AST heuristics only: no type checking, no test execution, no data-flow analysis.
- Only a bounded sample of the largest non-test source files is reviewed, so the score is an estimate.
- Score = 100 - (weighted issues per 1,000 LOC); severity weights (3/2/1) are a judgement call, not a standard.
- Cyclomatic complexity is approximated by counting branch nodes.
