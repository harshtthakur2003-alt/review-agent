# Code review report

**Goal:** Review the code quality of https://github.com/psf/requests and suggest fixes  
**Status:** `success`  
**Planner/LLM backend:** offline-heuristic

**Repository:** psf/requests (branch `main`)

## Result
- **Quality score:** 77.4 / 100
- Files analysed: 8 (3985 LOC)
- Issues: 56 (high 0, medium 34, low 22)

## Issues by type
| Issue | Count | Suggested fix |
|---|---|---|
| missing_docstring | 14 | Add short docstrings to public modules/classes/functions. |
| high_complexity | 14 | Split into smaller functions / early returns to reduce branching. |
| long_function | 12 | Extract cohesive blocks into helpers; aim for functions that fit on one screen. |
| todo_comment | 8 | Convert TODO/FIXME comments into tracked issues. |
| too_many_args | 4 | Group related parameters into a dataclass or config object. |
| swallowed_exception | 4 | Log the exception or re-raise; silent `pass` hides real bugs. |

## Top issues
| Severity | Location | Detail |
|---|---|---|
| medium | `src/requests/adapters.py:307` | cert_verify() cyclomatic complexity ~14 (limit 10) |
| medium | `src/requests/adapters.py:634` | send() is 115 lines (limit 60) |
| medium | `src/requests/adapters.py:634` | send() cyclomatic complexity ~20 (limit 10) |
| medium | `src/requests/auth.py:157` | build_digest_header() is 110 lines (limit 60) |
| medium | `src/requests/auth.py:157` | build_digest_header() cyclomatic complexity ~23 (limit 10) |
| medium | `src/requests/models.py:151` | _encode_params() cyclomatic complexity ~11 (limit 10) |
| medium | `src/requests/models.py:183` | _encode_files() is 69 lines (limit 60) |
| medium | `src/requests/models.py:183` | _encode_files() cyclomatic complexity ~21 (limit 10) |
| medium | `src/requests/models.py:323` | __init__() takes 11 parameters |
| medium | `src/requests/models.py:424` | prepare() takes 11 parameters |

## Hotspot files
- `src/requests/models.py`: 18 issues in 868 LOC
- `src/requests/sessions.py`: 10 issues in 641 LOC
- `src/requests/utils.py`: 10 issues in 832 LOC

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
| s2 | ok | 1 | - |
| s3 | ok | 1 | - |
| s4 | ok | 8 | - |
| s5 | ok | 8 | - |
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
