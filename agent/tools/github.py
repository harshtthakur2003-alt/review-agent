"""Repository access tools.

Repo contents come from a shallow `git clone` (no API quota, works for any git host, trivially testable
against local repos); GitHub's REST *search* endpoint is used only to repair misspelled repo names.
Everything is fetched live at run time - no dataset is bundled.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import requests

from ..models import ToolError
from .base import Tool, ToolRegistry

_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?$", re.I)
_SLUG_RE = re.compile(r"^([\w.-]+)/([\w.-]+)$")


def parse_repo(ref: str) -> tuple[str, str]:
    ref = (ref or "").strip()
    m = _URL_RE.match(ref) or _SLUG_RE.match(ref)
    if not m:
        raise ToolError("invalid_input", f"not a GitHub repo URL or owner/name slug: {ref!r}")
    return m.group(1), m.group(2)


class GitSource:
    """Runs git as a subprocess and maps its failure modes onto the agent's ToolError taxonomy."""

    NOT_FOUND = ("could not read username", "repository not found", "does not appear to be a git repository",
                 "not found in upstream", "remote branch", "couldn't find remote ref")
    TRANSIENT = ("could not resolve host", "unable to access", "early eof", "rpc failed", "timed out",
                 "connection reset", "429", "502", "503", "504", "unexpected disconnect")

    def __init__(self, base_url: str = "https://github.com", cache_dir: Path = Path(".cache/repos"),
                 search_api: str = "https://api.github.com", git_timeout: float = 120.0):
        self.base_url, self.cache_dir, self.search_api, self.git_timeout = base_url.rstrip("/"), Path(cache_dir), search_api, git_timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}

    # -- plumbing -------------------------------------------------------------------------
    def _git(self, *args: str, cwd: Optional[Path] = None) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
        try:
            p = subprocess.run(["git", *args], capture_output=True, text=True, timeout=self.git_timeout, env=env, cwd=cwd)
        except subprocess.TimeoutExpired:
            raise ToolError("transient", f"git {args[0]} timed out after {self.git_timeout}s")
        if p.returncode != 0:
            msg = (p.stderr or p.stdout).strip().lower()
            if any(s in msg for s in self.NOT_FOUND):
                raise ToolError("not_found", f"git {args[0]}: {msg[:140]}")
            if any(s in msg for s in self.TRANSIENT):
                raise ToolError("transient", f"git {args[0]}: {msg[:140]}")
            raise ToolError("fatal", f"git {args[0]} failed: {msg[:140]}")
        return p.stdout

    def url(self, owner: str, name: str) -> str:
        return f"{self.base_url}/{owner}/{name}"

    def _clone_dir(self, owner: str, name: str, branch: str) -> Path:
        return self.cache_dir / f"{owner}__{name}__{branch.replace('/', '_')}"

    # -- operations -----------------------------------------------------------------------
    def info(self, owner: str, name: str) -> dict:
        out = self._git("ls-remote", "--symref", self.url(owner, name), "HEAD")
        m = re.search(r"ref: refs/heads/(\S+)\s+HEAD", out)
        if not m:
            raise ToolError("invalid_output", "could not determine default branch from ls-remote")
        return {"full_name": f"{owner}/{name}", "default_branch": m.group(1), "url": self.url(owner, name)}

    def ensure_clone(self, owner: str, name: str, branch: str) -> Path:
        dest = self._clone_dir(owner, name, branch)
        lock = self._locks.setdefault(str(dest), threading.Lock())
        with lock:
            if (dest / ".git").exists():
                return dest
            tmp = Path(tempfile.mkdtemp(dir=self.cache_dir, prefix="clone_"))
            try:
                self._git("clone", "--depth", "1", "--single-branch", "--branch", branch, self.url(owner, name), str(tmp / "r"))
                shutil.rmtree(dest, ignore_errors=True)
                (tmp / "r").rename(dest)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return dest

    def search(self, query: str) -> list[dict]:
        try:
            r = requests.get(f"{self.search_api}/search/repositories", params={"q": query, "sort": "stars", "per_page": 30},
                             headers={"Accept": "application/vnd.github+json", "User-Agent": "repo-review-agent"}, timeout=15)
        except requests.RequestException as e:
            raise ToolError("transient", f"search request failed: {type(e).__name__}")
        if r.status_code in (403, 429):
            raise ToolError("rate_limited", f"search API {r.status_code}")
        if r.status_code >= 400:
            raise ToolError("fatal", f"search API {r.status_code}")
        return [{"full_name": i["full_name"], "stars": i["stargazers_count"]} for i in r.json().get("items", [])]


def _is_text(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\0" not in fh.read(2048)
    except OSError:
        return False


def build_github_tools(registry: ToolRegistry, src: GitSource) -> None:
    S = {"type": "string"}

    def repo_info(repo_url: str) -> dict:
        owner, name = parse_repo(repo_url)
        return src.info(owner, name)

    def search_repo(query: str) -> dict:
        return {"query": query, "candidates": src.search(query)}

    def _root(full_name: str, branch: Optional[str]) -> tuple[Path, str]:
        owner, name = parse_repo(full_name)
        branch = branch or src.info(owner, name)["default_branch"]
        return src.ensure_clone(owner, name, branch).resolve(), branch

    def list_files(full_name: str, branch: Optional[str] = None, glob: Optional[str] = None, limit: int = 200) -> dict:
        root, branch = _root(full_name, branch)
        rows = []
        for p in root.rglob("*"):
            rel = p.relative_to(root)
            if ".git" in rel.parts or not p.is_file():
                continue
            if glob and not fnmatch.fnmatch(str(rel), glob) and not fnmatch.fnmatch(rel.name, glob):
                continue
            rows.append({"path": str(rel), "size": p.stat().st_size})
        rows.sort(key=lambda r: r["path"])
        return {"branch": branch, "total_files": len(rows), "files": rows[: max(1, min(int(limit), 500))],
                "truncated": len(rows) > limit}

    def read_file(full_name: str, path: str, branch: Optional[str] = None, start_line: int = 1, max_lines: int = 250) -> dict:
        root, branch = _root(full_name, branch)
        target = (root / path).resolve()
        if root not in target.parents:
            raise ToolError("invalid_input", f"path escapes repository: {path}")
        if not target.is_file():
            raise ToolError("not_found", f"{path} not in {full_name}@{branch}")
        if not _is_text(target):
            raise ToolError("invalid_input", f"{path} looks like a binary file")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start, n = max(1, int(start_line)), max(1, min(int(max_lines), 600))
        chunk = lines[start - 1: start - 1 + n]
        return {"path": path, "total_lines": len(lines), "start_line": start,
                "text": "\n".join(f"{start + i}: {l}" for i, l in enumerate(chunk)),
                "has_more": start - 1 + n < len(lines)}

    def grep(full_name: str, pattern: str, branch: Optional[str] = None, glob: Optional[str] = None, max_matches: int = 40) -> dict:
        root, branch = _root(full_name, branch)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError("invalid_input", f"bad regex: {e}")
        hits = []
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if ".git" in rel.parts or not p.is_file() or p.stat().st_size > 1_000_000 or not _is_text(p):
                continue
            if glob and not fnmatch.fnmatch(str(rel), glob) and not fnmatch.fnmatch(rel.name, glob):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append({"path": str(rel), "line": i, "text": line.strip()[:160]})
                    if len(hits) >= max_matches:
                        return {"matches": hits, "truncated": True}
        return {"matches": hits, "truncated": False}

    def analyze_file(full_name: str, path: str, branch: Optional[str] = None) -> dict:
        from .analyzer import analyze_code
        text = read_file(full_name, path, branch, 1, 600)  # validates path, branch, binary-ness
        root, _ = _root(full_name, branch)
        return analyze_code(path, (root / path).read_text(encoding="utf-8", errors="replace"))

    reg = registry.register
    reg(Tool("repo_info", "Resolve a GitHub repo URL or 'owner/name' slug and return its default branch. "
             "Fails with not_found if the repository does not exist (check spelling; try repo_search).",
             repo_info, {"repo_url": {**S, "description": "https://github.com/owner/name or owner/name"}}, ["repo_url"],
             ["full_name", "default_branch"]))
    reg(Tool("repo_search", "Search GitHub repositories. Query 'user:<owner>' lists an owner's top repos "
             "(useful to repair a misspelled repository name).", search_repo, {"query": S}, ["query"], ["candidates"]))
    reg(Tool("repo_list_files", "Shallow-clone a repo and list its files (path, size). Optional glob like '*.py' or 'src/*'.",
             list_files, {"full_name": S, "branch": {**S, "description": "defaults to the default branch"}, "glob": S,
                          "limit": {"type": "integer"}}, ["full_name"], ["files"]))
    reg(Tool("repo_read_file", "Read a text file (line-numbered). Use start_line/max_lines to page through long files.",
             read_file, {"full_name": S, "path": S, "branch": S, "start_line": {"type": "integer"},
                         "max_lines": {"type": "integer"}}, ["full_name", "path"], ["text"]))
    reg(Tool("repo_grep", "Regex search across the repository's text files; returns path:line matches.",
             grep, {"full_name": S, "pattern": S, "branch": S, "glob": S, "max_matches": {"type": "integer"}},
             ["full_name", "pattern"], ["matches"]))
    reg(Tool("analyze_python_file", "Run a static AST analysis (complexity, bare except, mutable defaults, eval, ...) on ONE Python "
             "file in the repo. Findings are signals to interpret, not verdicts. Code is never executed.",
             analyze_file, {"full_name": S, "path": S, "branch": S}, ["full_name", "path"], ["issues", "loc"]))
