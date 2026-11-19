import pytest

from agent.models import ToolError
from agent.tools.analyzer import analyze_code, select_python_files, summarize_findings
from agent.tools.calculator import calculator
from agent.tools.github import parse_repo
from conftest import BAD, GOOD


def test_calculator_basic():
    assert calculator("round(100 - 1000 * 3 / max(40, 1), 1)")["value"] == 25.0


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "open('/etc/passwd')", "2 ** 999", "1/0", "1 +"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError) as e:
        calculator(expr)
    assert e.value.kind == "invalid_input"


def test_parse_repo_variants():
    assert parse_repo("https://github.com/psf/requests.git") == ("psf", "requests")
    assert parse_repo("github.com/psf/requests/tree/main") == ("psf", "requests")
    assert parse_repo("psf/requests") == ("psf", "requests")
    with pytest.raises(ToolError):
        parse_repo("not a repo")


def test_analyzer_finds_expected_issues():
    codes = {i["code"] for i in analyze_code("bad.py", BAD)["issues"]}
    assert {"bare_except", "mutable_default_arg", "eval_exec", "wildcard_import", "too_many_args",
            "swallowed_exception", "todo_comment", "high_complexity"} <= codes


def test_analyzer_clean_file_and_syntax_error():
    assert analyze_code("g.py", GOOD)["issues"] == []
    res = analyze_code("x.py", "def (:\n")
    assert res["issues"][0]["code"] == "syntax_error"


def test_select_skips_tests_and_tiny_files():
    files = [{"path": "src/a.py", "size": 5000}, {"path": "tests/test_a.py", "size": 9000},
             {"path": "src/tiny.py", "size": 10}, {"path": "docs/conf.py", "size": 3000}]
    assert [f["path"] for f in select_python_files(files)["files"]] == ["src/a.py"]
    with pytest.raises(ToolError):
        select_python_files([{"path": "tests/t.py", "size": 5000}])


def test_summarize_ranks_by_severity():
    s = summarize_findings([analyze_code("bad.py", BAD), analyze_code("g.py", GOOD)])
    assert s["by_severity"]["high"] >= 3 and s["worst_files"][0]["path"] == "bad.py"
    assert s["top_issues"][0]["severity"] == "high"
