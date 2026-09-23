"""
Project entry point for the Streamlit Medical RAG application.

Run:
    python main.py

Responsibilities:
1. Start Ollama in the background when it is installed but not already running.
2. Launch the Streamlit UI.
3. Keep model selection / installation blocking inside the Streamlit app.

The Streamlit app itself lives in:
    app/app.py
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = PROJECT_ROOT / "app" / "app.py"

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434


def command_exists(command: str) -> bool:
    """Return True when an executable is available on PATH."""
    return shutil.which(command) is not None


def port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check whether a local TCP service is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_ollama_if_available() -> subprocess.Popen | None:
    """
    Start `ollama serve` only when:
    - Ollama is installed, and
    - no Ollama service is already listening on port 11434.

    If Ollama is not installed, Streamlit is still launched.
    The UI will show the installation requirement and block access
    to the medical chat until Ollama becomes available.
    """
    if not command_exists("ollama"):
        print(
            "[Medical RAG] Ollama is not installed or is not on PATH.\n"
            "[Medical RAG] Starting the browser UI so installation "
            "instructions can be shown."
        )
        return None

    if port_is_open(OLLAMA_HOST, OLLAMA_PORT):
        print("[Medical RAG] Ollama server is already running.")
        return None

    print("[Medical RAG] Starting Ollama server...")

    creationflags = 0
    startupinfo = None

    # Prevent a second console window on Windows.
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )

    # Give Ollama a short period to bind to its local API port.
    for _ in range(20):
        if port_is_open(OLLAMA_HOST, OLLAMA_PORT):
            print("[Medical RAG] Ollama server started.")
            return process
        time.sleep(0.25)

    print(
        "[Medical RAG] Ollama was launched, but its API is not responding yet. "
        "The Streamlit app will continue checking it."
    )
    return process


def validate_project() -> None:
    """Fail early if the Streamlit application file is missing."""
    if not STREAMLIT_APP.exists():
        raise FileNotFoundError(
            "Streamlit app not found.\n"
            f"Expected: {STREAMLIT_APP}\n"
            "Create app/app.py before running the complete project."
        )


def launch_streamlit() -> int:
    """Launch Streamlit using the same Python interpreter as main.py."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
    ]

    print("[Medical RAG] Launching Streamlit...")
    print("[Medical RAG] Browser URL: http://127.0.0.1:8501")

    try:
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        return 0


def main() -> None:
    validate_project()

    if not command_exists("streamlit"):
        print(
            "Streamlit is not installed.\n"
            "Install project requirements first, for example:\n\n"
            "    pip install -r requirements.txt\n"
        )
        raise SystemExit(1)

    ollama_process = start_ollama_if_available()

    try:
        exit_code = launch_streamlit()
    finally:
        # Only terminate Ollama if THIS process started it.
        # If the user already had Ollama running, it is left untouched.
        if ollama_process is not None and ollama_process.poll() is None:
            print("\n[Medical RAG] Stopping Ollama server started by this app...")
            ollama_process.terminate()

            try:
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
