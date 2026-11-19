import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.tools import build_registry  # noqa: E402
from agent.tools.github import GitSource  # noqa: E402
from agent.trace import Tracer  # noqa: E402

GOOD = '"""Good module."""\n\n\ndef add(a, b):\n    """Add."""\n    return a + b\n' + "\n".join(f"# filler line {i}" for i in range(30)) + "\n"
BAD = '''"""Bad module."""
import os
from os.path import *


def risky(items=[], a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8):
    try:
        return eval("1+1")
    except:
        pass
    try:
        pass
    except ValueError:
        pass
    # TODO: fix this
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    for x in items:
                                        while x and a and b and c:
                                            return 1
    return 0
''' + "\n".join(f"# pad {i}" for i in range(60)) + "\n"


class LocalSource(GitSource):
    """GitSource against local directories; `search` is faked (no network)."""

    def __init__(self, root: Path, cache: Path, known=()):
        super().__init__(base_url=str(root), cache_dir=cache, git_timeout=30)
        self.known = list(known)

    def search(self, query):
        return [{"full_name": n, "stars": 1} for n in self.known]


def _git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd)})


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / "remotes"
    repo = root / "acme" / "widgets"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "good.py").write_text(GOOD * 1)
    (repo / "src" / "bad.py").write_text(BAD)
    (repo / "src" / "broken.py").write_text("def oops(:\n" + "x = 1\n" * 200)
    (repo / "tests" / "test_x.py").write_text(BAD)
    _git(repo, "init", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return root


@pytest.fixture
def make_env(remote, tmp_path):
    def _make(known=("acme/widgets",)):
        src = LocalSource(remote, tmp_path / "cache", known)
        return build_registry(src, timeout=30), Tracer(quiet=True)
    return _make
