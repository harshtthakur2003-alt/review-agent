"""Streamlit UI for the review agent.

    pip install streamlit requests
    streamlit run app.py

The UI is only a front-end: it calls the same `run_agent` loop as the CLI, streams the agent's trace live,
and renders the model-authored report. It also lets you browse saved runs in examples/ and runs/.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import requests
import streamlit as st

from agent.faults import inject_faults
from agent.llm import make_llm
from agent.loop import run_agent
from agent.models import LLMError
from agent.report import to_markdown
from agent.tools import GitSource, build_registry
from agent.trace import Tracer

ROOT = Path(__file__).parent
KIND_STYLE = {
    "plan": ("🗺️", "blue"), "step": ("👣", "gray"), "call": ("🔧", "violet"), "result": ("✅", "green"),
    "error": ("❌", "red"), "recover": ("🩹", "orange"), "skip": ("⏭️", "gray"), "info": ("ℹ️", "gray"),
    "report": ("🏁", "green"), "think": ("💭", "blue"),
}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

st.set_page_config(page_title="Review Agent", page_icon="🔎", layout="wide")


class StreamlitTracer(Tracer):
    """Tracer that also renders every event live in the page."""

    def __init__(self, jsonl_path: Path, container):
        super().__init__(jsonl_path, quiet=True)
        self.container = container

    def emit(self, kind: str, message: str, **data) -> None:
        super().emit(kind, message, **data)
        icon, color = KIND_STYLE.get(kind, ("•", "gray"))
        t = self.events[-1]["t"]
        self.container.markdown(f"`{t:6.2f}s` {icon} :{color}[**{kind.upper()}**] {message}")


def render_report(rep: dict) -> None:
    status = rep.get("status", "?")
    if status == "needs_clarification":
        st.warning(f"**The agent asked for clarification instead of guessing:**\n\n{rep.get('question', '')}")
        return
    if "findings" not in rep:
        st.error(f"Run failed: {rep.get('error', 'unknown error')}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Quality score", f"{rep.get('quality_score', '–')}/100")
    c3.metric("Findings", len(rep.get("findings", [])))
    c4.metric("Recoveries", len(rep.get("recoveries", [])))
    st.markdown(f"**Repository:** `{rep.get('repository', '')}`")
    st.info(rep.get("summary", ""))
    with st.expander("Score rationale"):
        st.write(rep.get("score_rationale", ""))

    st.subheader("Findings")
    for f in rep.get("findings", []):
        sev = f.get("severity", "low")
        with st.expander(f"{SEV_ICON.get(sev, '•')} [{sev}] {f.get('title', '')} — `{f.get('location', '')}`"):
            st.markdown(f"**Evidence**\n\n{f.get('evidence', '')}")
            st.markdown(f"**Suggested fix**\n\n{f.get('suggested_fix', '')}")

    left, right = st.columns(2)
    with left:
        st.subheader("What the agent did")
        for s in rep.get("what_i_did", []):
            st.markdown(f"- {s}")
        st.subheader("Assumptions")
        for s in rep.get("assumptions", []) or ["(none)"]:
            st.markdown(f"- {s}")
    with right:
        st.subheader("Recoveries")
        recs = rep.get("recoveries", [])
        if not recs:
            st.caption("No failures hit during this run.")
        for r in recs:
            st.markdown(f"- **Problem:** {r.get('problem', '')}  \n  **Action:** {r.get('action', '')}")
        st.subheader("Limitations")
        for s in rep.get("limitations", []) or ["(none)"]:
            st.markdown(f"- {s}")
    if rep.get("run"):
        with st.expander("Run metadata"):
            st.json(rep["run"])


def render_trace(events: list[dict]) -> None:
    for ev in events:
        icon, color = KIND_STYLE.get(ev.get("kind"), ("•", "gray"))
        st.markdown(f"`{ev.get('t', 0):6.2f}s` {icon} :{color}[**{str(ev.get('kind', '')).upper()}**] {ev.get('message', '')}")


# ---------------------------------------------------------------- sidebar: model settings
with st.sidebar:
    st.header("⚙️ Model")
    provider = st.selectbox("Provider", ["openai (OpenAI-compatible: Gemini, Groq, Ollama…)", "anthropic"])
    provider = provider.split(" ")[0]
    if provider == "openai":
        base_url = st.text_input("Base URL", os.environ.get("OPENAI_BASE_URL",
                                 "https://generativelanguage.googleapis.com/v1beta/openai"))
        key = st.text_input("API key", os.environ.get("OPENAI_API_KEY", ""), type="password")
        if st.button("🔄 Fetch models available to this key", disabled=not key):
            try:
                r = requests.get(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
                r.raise_for_status()
                ids = sorted(m.get("id", "").removeprefix("models/") for m in r.json().get("data", []))
                st.session_state["models"] = [i for i in ids if i]
            except Exception as e:  # show the provider's message instead of crashing the UI
                st.session_state["models"] = []
                st.error(f"Could not list models: {e}")
        models = st.session_state.get("models") or []
        if models:
            model = st.selectbox("Model", models)
        else:
            model = st.text_input("Model", os.environ.get("AGENT_MODEL", ""),
                                  help="Click 'Fetch models' above to pick from the models your key can use")
    else:
        base_url = ""
        model = st.text_input("Model", os.environ.get("AGENT_MODEL", "claude-sonnet-5-5"))
        key = st.text_input("API key", os.environ.get("ANTHROPIC_API_KEY", ""), type="password")
    st.caption("The key stays in this local process's environment; it is never written to disk.")

    st.header("🧪 Robustness")
    faults = st.text_area("Inject faults (one TOOL:MODE[:N] per line)", "",
                          placeholder="repo_list_files:timeout:2\nrepo_read_file:malformed:1\nrepo_read_file:rate_limit:1")
    max_turns = st.slider("Max model turns", 5, 60, 30)
    backoff = st.number_input("Retry backoff (s)", 0.0, 10.0, 1.0, 0.5)
    context_chars = st.number_input("Context budget (chars)", 4000, 400000, 60000, 2000,
                                    help="Older tool outputs are compacted above this. Use ~12000 on free tiers "
                                         "with small tokens-per-minute limits (e.g. Groq).")

st.title("🔎 Autonomous Code Review Agent")
st.caption("An LLM plans, chooses tools, recovers from failures and writes the report. No canned answers.")

tab_run, tab_browse = st.tabs(["▶️ Run the agent", "📂 Saved runs"])

# ---------------------------------------------------------------- tab 1: live run
with tab_run:
    goal = st.text_input("Goal", "Review the code quality of https://github.com/pallets/markupsafe and suggest fixes")
    ex = st.columns(3)
    if ex[0].button("Example: normal"):
        st.session_state["goal_hint"] = "Review the code quality of https://github.com/psf/requests and suggest fixes"
    if ex[1].button("Example: misspelled repo"):
        st.session_state["goal_hint"] = "Review the code quality of github.com/psf/requets and suggest fixes"
    if ex[2].button("Example: ambiguous goal"):
        st.session_state["goal_hint"] = "make my code better"
    if st.session_state.get("goal_hint"):
        st.caption(f"Copy into the goal box: `{st.session_state['goal_hint']}`")

    if st.button("Run agent", type="primary", disabled=not goal.strip()):
        if not key:
            st.error("Enter an API key in the sidebar first.")
            st.stop()
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = key
            os.environ["OPENAI_BASE_URL"] = base_url
        else:
            os.environ["ANTHROPIC_API_KEY"] = key
        try:
            llm = make_llm(provider, model or None)
        except LLMError as e:
            st.error(str(e))
            st.stop()

        out = ROOT / "runs" / "streamlit"
        out.mkdir(parents=True, exist_ok=True)
        st.subheader("Live trace")
        trace_box = st.container(height=420, border=True)
        tracer = StreamlitTracer(out / "trace.jsonl", trace_box)
        registry = build_registry(GitSource(cache_dir=Path(tempfile.mkdtemp(prefix="repo-agent-"))), 90.0)
        specs = [s.strip() for s in faults.splitlines() if s.strip()]
        try:
            if specs:
                inject_faults(registry, specs)
        except ValueError as e:
            st.error(f"Bad fault spec: {e}")
            st.stop()
        with st.spinner("Agent is working…"):
            report = run_agent(goal, llm, registry, tracer, max_turns=max_turns, backoff=backoff, tool_timeout=90.0,
                               context_chars=int(context_chars))
        tracer.close()
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        md = (to_markdown(report) if "findings" in report or report.get("status") == "needs_clarification"
              else f"# Run failed\n\n{report.get('error')}\n")
        (out / "report.md").write_text(md, encoding="utf-8")
        (out / "transcript.txt").write_text("\n".join(tracer.lines) + "\n", encoding="utf-8")

        st.divider()
        st.subheader("Report")
        render_report(report)
        st.download_button("Download report.md", md, file_name="report.md")
        st.download_button("Download report.json", json.dumps(report, indent=2, default=str), file_name="report.json")

# ---------------------------------------------------------------- tab 2: browse saved runs
with tab_browse:
    run_dirs = sorted({p.parent for base in ("examples", "runs") for p in (ROOT / base).glob("**/report.json")})
    if not run_dirs:
        st.info("No saved runs yet. Generate some with the CLI (see README) or the Run tab.")
    else:
        choice = st.selectbox("Run", run_dirs, format_func=lambda p: str(p.relative_to(ROOT)))
        rep = json.loads((choice / "report.json").read_text(encoding="utf-8"))
        sub_report, sub_trace = st.tabs(["Report", "Trace"])
        with sub_report:
            render_report(rep)
        with sub_trace:
            tf = choice / "trace.jsonl"
            if tf.exists():
                render_trace([json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()])
            else:
                st.caption("No trace.jsonl in this run.")
