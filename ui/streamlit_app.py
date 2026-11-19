import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("POLICYPILOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_TIMEOUT_SECONDS = (
    180  # the agent may search up to 3 times and call Gemini several times
)

STATUS_LABELS = {
    "answered": ("✅", "Answered from policy text"),
    "clarify": ("❓", "More detail needed"),
    "abstained": ("⚠️", "No fully supported answer"),
}

EXAMPLE_QUESTIONS = [
    "Can I work from home while on probation?",
    "How many days per week can I work remotely?",
    "What are the core hours for remote employees?",
]


@st.cache_resource
def http_session() -> requests.Session:
    session = requests.Session()
    # Do not route requests to the local API through a corporate proxy
    # (HTTP_PROXY / HTTPS_PROXY variables). Change this if the API is remote
    # and must be reached through the proxy.
    session.trust_env = False
    return session


def check_health() -> tuple[bool, str]:
    try:
        response = http_session().get(f"{API_URL}/api/v1/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        return (
            True,
            f"{data.get('service')} {data.get('version')} ({data.get('environment')})",
        )
    except requests.RequestException as exc:
        return False, type(exc).__name__


def ask_api(question: str, search_mode: str) -> dict[str, Any]:
    """Return {'result': ...} on success or {'error': ...} with a readable message."""
    try:
        response = http_session().post(
            f"{API_URL}/api/v1/ask",
            json={"question": question, "search_mode": search_mode},
            timeout=ASK_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return {
            "error": f"The API did not respond within {ASK_TIMEOUT_SECONDS} seconds."
        }
    except requests.ConnectionError:
        return {"error": f"Cannot reach the API at {API_URL}. Is it running?"}

    if response.ok:
        return {"result": response.json()}
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text[:300]
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return {"error": f"API error {response.status_code}: {detail}"}


def format_pages(citation: dict[str, Any]) -> str:
    start, end = citation.get("page_start"), citation.get("page_end")
    if start is None:
        return "no page info"
    return f"p. {start}" if start == end or end is None else f"pp. {start}-{end}"


def render_result(result: dict[str, Any], show_trace: bool) -> None:
    icon, label = STATUS_LABELS.get(result["status"], ("•", result["status"]))
    st.caption(f"{icon} {label}")
    st.markdown(result["answer"])

    citations = result.get("citations", [])
    if citations:
        with st.expander(f"Sources ({len(citations)})", expanded=True):
            for c in citations:
                st.markdown(
                    f"**[{c['evidence_id']}]** {c.get('title') or c['policy_id']} "
                    f"({c['policy_id']} v{c['version']}) · §{c['section_number']} "
                    f"{c.get('heading') or ''} · {c.get('source_file') or 'unknown file'}, {format_pages(c)}"
                )
    if show_trace and result.get("trace"):
        with st.expander(
            f"Agent steps ({result.get('retrieval_attempts', 0)} search attempt(s))"
        ):
            st.code("\n".join(result["trace"]), language=None)


def render_message(message: dict[str, Any], show_trace: bool) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "error" in message:
            st.error(message["error"])
        else:
            render_result(message["result"], show_trace)


def main() -> None:
    st.set_page_config(page_title="PolicyPilot", page_icon="📘", layout="centered")
    st.session_state.setdefault("messages", [])

    with st.sidebar:
        st.header("PolicyPilot")
        healthy, info = check_health()
        if healthy:
            st.success(f"API connected: {info}")
        else:
            st.error(f"API not reachable at {API_URL} ({info})")
        search_mode = st.selectbox(
            "Search mode",
            ["hybrid", "semantic", "keyword"],
            help="Hybrid combines meaning-based and exact-term search.",
        )
        show_trace = st.toggle("Show agent steps", value=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption(
            "Each question is answered independently. Answers come only from indexed "
            "policies (synthetic data). Confirm important decisions with HR."
        )

    st.title("📘 PolicyPilot")
    st.write(
        "Ask about HR and compliance policies. Every answer cites the clause it is based on."
    )

    for message in st.session_state.messages:
        render_message(message, show_trace)

    question = st.chat_input("Ask a policy question")
    if not st.session_state.messages and question is None:
        st.write("Try an example:")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"example-{example}"):
                question = example

    if question and question.strip():
        user_message = {"role": "user", "content": question.strip()}
        st.session_state.messages.append(user_message)
        render_message(user_message, show_trace)
        with st.spinner("Searching policies and checking evidence..."):
            reply = ask_api(question.strip(), search_mode)
        assistant_message = {"role": "assistant", **reply}
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message, show_trace)


main()
