"""Repository access tools.

Repo contents come from a shallow `git clone` (no API quota, works for any git host, trivially testable
against local repos); GitHub's REST *search* endpoint is used only to repair misspelled repo names.
Everything is fetched live at run time - no dataset is bundled.
"""
from __future__ import annotations

import difflib
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


def build_github_tools(registry: ToolRegistry, src: GitSource) -> None:
    def repo_info(repo_url: str) -> dict:
        owner, name = parse_repo(repo_url)
        return src.info(owner, name)

    def search_repo(query: str) -> dict:
        return {"query": query, "candidates": src.search(query)}

    def list_files(full_name: str, branch: str) -> dict:
        owner, name = parse_repo(full_name)
        root = src.ensure_clone(owner, name, branch)
        files = [{"path": str(p.relative_to(root)), "size": p.stat().st_size}
                 for p in sorted(root.rglob("*.py")) if ".git" not in p.relative_to(root).parts and p.is_file()]
        return {"files": files, "branch": branch}

    def read_file(full_name: str, path: str, ref: Optional[str] = None) -> dict:
        owner, name = parse_repo(full_name)
        branch = ref or src.info(owner, name)["default_branch"]
        root = src.ensure_clone(owner, name, branch).resolve()
        target = (root / path).resolve()
        if root not in target.parents:
            raise ToolError("invalid_input", f"path escapes repository: {path}")
        if not target.is_file():
            raise ToolError("not_found", f"{path} not in {full_name}@{branch}")
        return {"path": path, "text": target.read_text(encoding="utf-8", errors="replace")}

    # ---- fallbacks: deterministic self-correction strategies -------------------------------
    def repo_info_fallback(args: dict, err: ToolError, reg: ToolRegistry):
        """Typo'd / ambiguous repo name -> search GitHub, accept only a close, unambiguous match."""
        if err.kind != "not_found":
            return None
        owner, name = parse_repo(args["repo_url"])
        # GitHub search is literal (a typo matches nothing), so list the owner's repos and fuzzy-match names.
        cands = reg.call("repo_search", {"query": f"user:{owner}"})["candidates"]
        target = f"{owner}/{name}".lower()
        scored = sorted(((difflib.SequenceMatcher(None, target, c["full_name"].lower()).ratio(), c) for c in cands),
                        key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.85 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
            best = scored[0][1]["full_name"]
            return {"repo_url": best}, (f"'{owner}/{name}' not found; searched the owner's repositories and the closest unambiguous match is "
                                        f"'{best}' (similarity {scored[0][0]:.2f}) - proceeding with it, flagged as an assumption")
        return None

    def list_files_fallback(args: dict, err: ToolError, reg: ToolRegistry):
        if err.kind != "not_found":
            return None
        for alt in ("main", "master"):
            if alt != args["branch"]:
                return {"full_name": args["full_name"], "branch": alt}, f"branch '{args['branch']}' not found; trying '{alt}'"
        return None

    registry.register(Tool("repo_info", "Resolve a GitHub repo URL/slug; returns default branch. Fails if repo doesn't exist.",
                           repo_info, ["repo_url"], ["full_name", "default_branch"], repo_info_fallback))
    registry.register(Tool("repo_search", "Search GitHub repositories (e.g. 'user:<owner>' lists an owner's top repos; used to repair misspelled repos).",
                           search_repo, ["query"], ["candidates"]))
    registry.register(Tool("repo_list_files", "Shallow-clone a repo branch and list its Python files.", list_files,
                           ["full_name", "branch"], ["files"], list_files_fallback))
    registry.register(Tool("repo_read_file", "Read one file's text from a cloned repo.", read_file,
                           ["full_name", "path"], ["text"]))
