"""
Chatbot page — free-form chat about the BOLT / AT-CBGRU project.

Grounded with a factual summary of this run's actual results (leaderboard,
DM significance, ablation, anomaly benchmark) so answers cite real numbers
instead of hallucinating them.

Runs on Llama via Groq's free API. The key is set by YOU (the app owner),
not typed in by whoever opens the app. Set it one of two ways before
running:

  1. Streamlit secrets (recommended, works locally and on Streamlit
     Community Cloud): create `.streamlit/secrets.toml` next to app.py with:
         GROQ_API_KEY = "your-key-here"

  2. Environment variable:
         export GROQ_API_KEY="your-key-here"     (macOS/Linux)
         set GROQ_API_KEY=your-key-here           (Windows cmd)

Get a free key at https://console.groq.com/keys — it starts with "gsk_".
"""

import os
import time

import streamlit as st

from utils import build_project_context

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")
st.title("💬 Ask about this project")

MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 3  # 1 initial attempt + 2 retries
RETRY_DELAY_SECONDS = 2  # doubles each retry: 2s, then 4s


def _is_retryable(exc: Exception) -> bool:
    """503/502 (service unavailable) and 429 (rate limit) are transient —
    worth retrying. Auth/permission errors (401, 403) are not — retrying
    just wastes time."""
    msg = str(exc)
    return any(code in msg for code in ("503", "502", "UNAVAILABLE", "429", "rate_limit"))


SUGGESTED_QUESTIONS = [
    "Which model performed best and by how much?",
    "Is the proposed model significantly better than BiLSTM?",
    "What does the ablation study show is most important?",
    "How good is the anomaly detection, honestly?",
    "What data was this trained and tested on?",
]


def _clean(key: str) -> str:
    """Strips whitespace and accidental surrounding quotes — e.g. Windows
    cmd's `set VAR="value"` keeps the quote characters as part of the
    value (unlike bash), which silently produces an invalid key."""
    key = key.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        key = key[1:-1].strip()
    return key


def _get_api_key() -> tuple[str, str]:
    """Returns (key, source_label) so the sidebar can show where it came from."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return _clean(st.secrets["GROQ_API_KEY"]), "secrets.toml"
    except Exception:
        pass
    env_key = os.environ.get("GROQ_API_KEY", "")
    if env_key:
        return _clean(env_key), "environment variable"
    return "", ""


api_key, key_source = _get_api_key()

with st.sidebar:
    st.caption(f"Model: `{MODEL}` (via Groq)")
    if api_key:
        masked = f"...{api_key[-4:]}" if len(api_key) > 4 else "(very short — likely wrong)"
        prefix_ok = api_key.startswith("gsk_")
        st.caption(
            f"Key: `{masked}` ({len(api_key)} chars) from {key_source}"
            + ("" if prefix_ok else " — ⚠️ doesn't start with `gsk_`, wrong key type?")
        )
    if st.button("Clear chat history"):
        st.session_state["chat_messages"] = []
        st.rerun()

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if not api_key:
    st.warning(
        "No Groq API key configured for this app. Set `GROQ_API_KEY` in "
        "`.streamlit/secrets.toml` or as an environment variable before "
        "running — see the comment at the top of this file for both options. "
        "Get a free key at https://console.groq.com/keys (it starts with `gsk_`)."
    )
    st.stop()

try:
    from groq import Groq
except ImportError:
    st.error(
        "The `groq` package isn't installed. Add `groq` to "
        "requirements.txt and `pip install -r requirements.txt`."
    )
    st.stop()

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the BOLT / "
    "AT-CBGRU energy forecasting and anomaly detection project below. "
    "Use ONLY the facts in this summary — if something isn't covered here, "
    "say you don't have that in the run's exported results rather than "
    "guessing. Be concise and precise with numbers.\n\n"
    + build_project_context()
)


def _ask(question: str):
    """Sends `question` to Llama (via Groq), streaming the reply into the chat."""
    st.session_state["chat_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        client = Groq(api_key=api_key)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["chat_messages"]
        ]

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            full_text = ""
            last_error = None
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_text += delta
                        placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                break  # success — no retry needed
            except Exception as e:
                last_error = e
                if _is_retryable(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    placeholder.markdown(
                        f"⏳ Model is busy (attempt {attempt}/{MAX_RETRIES}) — "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                break  # not retryable, or out of retries

        if last_error is not None:
            if _is_retryable(last_error):
                full_text = (
                    f"Llama is experiencing high demand right now and didn't respond "
                    f"after {MAX_RETRIES} attempts. This is temporary on Groq's side — "
                    f"please try again in a minute.\n\n*Details: {last_error}*"
                )
            else:
                full_text = f"API call failed: {last_error}"
            placeholder.markdown(full_text)

    st.session_state["chat_messages"].append({"role": "assistant", "content": full_text})


# Replay history so far.
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Suggested-question chips: shown until the first message, so there's an
# easy on-ramp, but they never block typing a free-form question instead.
if not st.session_state["chat_messages"]:
    st.caption("Try asking:")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(q, key=f"suggest_{q}", width="stretch"):
                st.session_state["pending_prompt"] = q
                st.rerun()

prompt = st.chat_input("Ask about the leaderboard, ablation, anomaly results, dataset...")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    _ask(prompt)