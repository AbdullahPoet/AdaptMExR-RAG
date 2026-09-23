"""
Streamlit UI for the Local Medical RAG application.

Save as:
    app/app.py

Launch the entire project with:
    python main.py

Startup flow:
1. Check Ollama installation.
2. Check Ollama server/version.
3. Discover locally installed Ollama models.
4. Require the user to choose one model.
5. Require a Tavily API key and validate it.
6. Block access to the medical chat until setup is complete.
7. Start a ChatGPT-style medical chat with session memory.
"""

from __future__ import annotations

import shutil
from typing import Any

import requests
import streamlit as st


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

LOCAL_REQUEST_TIMEOUT = 3
TAVILY_REQUEST_TIMEOUT = 12


st.set_page_config(
    page_title="Local Medical RAG",
    page_icon="🩺",
    layout="centered",
)


# =====================================================================
# Ollama helpers
# =====================================================================

def ollama_command_exists() -> bool:
    """Return True if the Ollama executable is available on PATH."""
    return shutil.which("ollama") is not None


def get_ollama_version() -> str | None:
    """Return the running Ollama server version."""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/version",
            timeout=LOCAL_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        version = data.get("version")

        return str(version) if version else None

    except (requests.RequestException, ValueError):
        return None


def get_installed_models() -> list[str]:
    """
    Discover locally installed Ollama models.

    Example:
        qwen3:8b
        llama3.1:8b
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=LOCAL_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        names: list[str] = []

        for model in models:
            name = model.get("name") or model.get("model")

            if name:
                names.append(str(name))

        return sorted(set(names))

    except (requests.RequestException, ValueError):
        return []


# =====================================================================
# Tavily helpers
# =====================================================================

def validate_tavily_api_key(
    api_key: str,
) -> tuple[bool, str]:
    """
    Validate the Tavily API key with a minimal live search request.

    The key is never written to disk. It is kept only in Streamlit's
    in-memory session state for the current browser session.
    """

    api_key = str(api_key).strip()

    if not api_key:
        return False, "Tavily API key is empty."

    payload = {
        "api_key": api_key,
        "query": "medical evidence",
        "search_depth": "basic",
        "max_results": 1,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            json=payload,
            timeout=TAVILY_REQUEST_TIMEOUT,
        )

    except requests.ConnectionError:
        return (
            False,
            "Could not connect to Tavily. Check your internet connection.",
        )

    except requests.Timeout:
        return (
            False,
            "Tavily validation timed out. Please try again.",
        )

    except requests.RequestException as exc:
        return (
            False,
            f"Tavily validation failed: {exc}",
        )

    if response.status_code in {401, 403}:
        return (
            False,
            "The Tavily API key was rejected.",
        )

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:300]

        return (
            False,
            f"Tavily returned HTTP {response.status_code}: {detail}",
        )

    try:
        data = response.json()
    except ValueError:
        return (
            False,
            "Tavily returned an invalid response.",
        )

    # A successful Tavily search response should normally contain results.
    if "results" not in data:
        return (
            False,
            "Tavily responded, but the response format was unexpected.",
        )

    return True, "Tavily connection verified."


# =====================================================================
# Session state
# =====================================================================

def initialize_session_state() -> None:
    """Initialize application state once per browser session."""

    defaults: dict[str, Any] = {
        "setup_complete": False,
        "selected_model": None,
        "tavily_api_key": None,
        "tavily_verified": False,
        "messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """Clear conversation history but keep model/API-key setup."""
    st.session_state.messages = []


def reset_setup() -> None:
    """
    Return to the setup screen.

    Clear the API key from memory as part of the reset.
    """
    st.session_state.setup_complete = False
    st.session_state.selected_model = None
    st.session_state.tavily_api_key = None
    st.session_state.tavily_verified = False
    st.session_state.messages = []


# =====================================================================
# RAG pipeline
# =====================================================================

@st.cache_resource(show_spinner="Loading medical RAG pipeline...")
def load_rag_pipeline(
    model_name: str,
    tavily_api_key: str,
):
    """
    Load one cached pipeline for the selected Ollama model/API-key pair.

    The next pipeline revision will consume `tavily_api_key` directly for
    UPDATED_NEED queries. For backward compatibility with the current pipeline,
    the key is also attached to the object after construction.
    """

    from rag.pipeline import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline(
        ollama_model=model_name,
        ollama_base_url=OLLAMA_BASE_URL,
        tavily_api_key=tavily_api_key,
    )

    return pipeline


# =====================================================================
# Setup screen
# =====================================================================

def render_setup_screen() -> None:
    st.title("🩺 Local Medical RAG")
    st.caption(
        "Local Ollama medical assistant with MedCPT retrieval "
        "and Tavily for current-information queries."
    )

    st.subheader("System setup")

    # -----------------------------------------------------------------
    # Step 1 — Ollama installation
    # -----------------------------------------------------------------

    st.markdown("### 1. Ollama")

    if not ollama_command_exists():
        st.error(
            "Ollama is not installed or is not available on your PATH."
        )

        st.markdown(
            """
Install Ollama first.

After installation, verify it from a terminal:

```bash
ollama --version
```

Then restart:

```bash
python main.py
```
"""
        )

        st.stop()

    version = get_ollama_version()

    if version is None:
        st.error(
            "Ollama is installed, but the local Ollama server "
            "is not responding."
        )

        st.markdown(
            """
`main.py` normally starts Ollama automatically.

If necessary, start it manually:

```bash
ollama serve
```

Then click **Check Ollama again**.
"""
        )

        if st.button(
            "Check Ollama again",
            type="primary",
        ):
            st.rerun()

        st.stop()

    st.success(f"Ollama available — version {version}")

    # -----------------------------------------------------------------
    # Step 2 — local model selection
    # -----------------------------------------------------------------

    st.markdown("### 2. Choose an Ollama model")

    models = get_installed_models()

    if not models:
        st.warning(
            "Ollama is running, but no local LLM is installed."
        )

        st.markdown(
            """
Pull at least one model, for example:

```bash
ollama pull qwen3:8b
```

After it finishes, click **Check models again**.
"""
        )

        if st.button(
            "Check models again",
            type="primary",
        ):
            st.rerun()

        st.stop()

    selected_model = st.selectbox(
        "Installed models",
        options=models,
        index=None,
        placeholder="Select an Ollama model...",
    )

    # -----------------------------------------------------------------
    # Step 3 — Tavily key
    # -----------------------------------------------------------------

    st.markdown("### 3. Tavily API key")

    st.caption(
        "Tavily is used only when the router determines that a question "
        "requires current or recently updated information."
    )

    tavily_key_input = st.text_input(
        "Enter Tavily API key",
        type="password",
        placeholder="tvly-...",
        help=(
            "The key is kept in this Streamlit session only and "
            "is not written to a project file."
        ),
    )

    if st.button(
        "Verify Tavily key",
        disabled=not bool(tavily_key_input.strip()),
    ):
        with st.spinner("Verifying Tavily API key..."):
            valid, message = validate_tavily_api_key(
                tavily_key_input
            )

        if valid:
            st.session_state.tavily_api_key = (
                tavily_key_input.strip()
            )
            st.session_state.tavily_verified = True
            st.success(message)
        else:
            st.session_state.tavily_api_key = None
            st.session_state.tavily_verified = False
            st.error(message)

    # Persist verification feedback across Streamlit reruns.
    if st.session_state.tavily_verified:
        st.success("Tavily API key verified.")

    # -----------------------------------------------------------------
    # Step 4 — continue
    # -----------------------------------------------------------------

    st.markdown("### 4. Start")

    setup_ready = (
        selected_model is not None
        and st.session_state.tavily_verified
        and bool(st.session_state.tavily_api_key)
    )

    if selected_model is None:
        st.info("Choose an Ollama model.")

    if not st.session_state.tavily_verified:
        st.info("Verify a Tavily API key.")

    if st.button(
        "Continue to Medical Assistant",
        type="primary",
        use_container_width=True,
        disabled=not setup_ready,
    ):
        st.session_state.selected_model = selected_model
        st.session_state.setup_complete = True
        st.session_state.messages = []

        st.rerun()

    # Strict blocking: the chat UI cannot render before setup is complete.
    st.stop()


# =====================================================================
# Chat UI
# =====================================================================

def render_chat_screen() -> None:
    model_name = st.session_state.selected_model
    tavily_api_key = st.session_state.tavily_api_key

    st.title("🩺 Medical Assistant")

    st.caption(
        f"Local RAG • Ollama model: `{model_name}` • "
        "Tavily current-information routing enabled"
    )

    # -----------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------

    with st.sidebar:
        st.header("Session")

        st.write("Ollama model")
        st.code(model_name)

        st.write("Tavily")
        st.success("Connected")

        st.divider()

        if st.button(
            "New chat",
            use_container_width=True,
        ):
            reset_chat()
            st.rerun()

        if st.button(
            "Change setup",
            use_container_width=True,
        ):
            reset_setup()

            # Cached pipeline may contain the old API key.
            load_rag_pipeline.clear()

            st.rerun()

        st.divider()

        st.caption(
            "This assistant provides informational medical guidance "
            "using the configured evidence sources. For emergencies, "
            "seek immediate professional care."
        )

    # -----------------------------------------------------------------
    # Replay chat memory
    # -----------------------------------------------------------------

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------------------------------------------
    # New message
    # -----------------------------------------------------------------

    question = st.chat_input(
        "Ask a medical question..."
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Do not include the just-added question twice.
    conversation_history = (
        st.session_state.messages[:-1]
    )

    # -----------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------

    try:
        rag_pipeline = load_rag_pipeline(
            model_name=model_name,
            tavily_api_key=tavily_api_key,
        )

    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(
                f"Could not initialize the Medical RAG pipeline: {exc}"
            )
        return

    # -----------------------------------------------------------------
    # Generate answer
    # -----------------------------------------------------------------

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching evidence and generating answer..."
        ):
            try:
                result = rag_pipeline.ask(
                    question=question,
                    chat_history=conversation_history,
                )

                if isinstance(result, dict):
                    answer = str(
                        result.get("answer") or ""
                    ).strip()
                else:
                    answer = str(result).strip()

                if not answer:
                    answer = (
                        "The pipeline completed but returned "
                        "an empty response."
                    )

                st.markdown(answer)

            except Exception as exc:
                st.error(
                    f"Medical RAG request failed: {exc}"
                )
                return

    # -----------------------------------------------------------------
    # Store assistant response as chat memory
    # -----------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# =====================================================================
# Application entry point
# =====================================================================

def main() -> None:
    initialize_session_state()

    if not st.session_state.setup_complete:
        render_setup_screen()

    # Defensive check: never enter chat without both dependencies.
    if (
        not st.session_state.selected_model
        or not st.session_state.tavily_verified
        or not st.session_state.tavily_api_key
    ):
        reset_setup()
        st.rerun()

    render_chat_screen()


if __name__ == "__main__":
    main()
