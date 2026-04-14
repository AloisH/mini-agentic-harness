#!/usr/bin/env python3
"""Mini Agentic Harness — a tiny Claude Code-like loop for local LLMs via LM Studio."""

import re
import subprocess
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"          # LM Studio ignores this but the client needs it
MODEL = "local-model"          # LM Studio ignores this too
MAX_ITERATIONS = 20
BASH_TIMEOUT = 30              # seconds per command

SYSTEM_PROMPT = """\
You are a helpful AI assistant that can run bash commands on the user's machine.

When you need to execute a shell command, wrap it like this:

<bash>
your command here
</bash>

Rules:
- You can use multiple <bash> blocks in one reply.
- After each bash block is executed you will receive the output inside <bash_result> tags.
- Keep reasoning about the results and continue using bash until the task is done.
- When you have a final answer, reply normally with NO <bash> blocks.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASH_RE = re.compile(r"<bash>\s*(.*?)\s*</bash>", re.DOTALL)
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_CYAN   = "\033[36m"
ANSI_GREEN  = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED    = "\033[31m"
ANSI_DIM    = "\033[2m"


def _c(color: str, text: str) -> str:
    """Wrap text in an ANSI color if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{color}{text}{ANSI_RESET}"
    return text


def run_bash(command: str) -> str:
    """Execute a shell command and return combined stdout+stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        return "\n".join(parts).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {BASH_TIMEOUT}s"
    except Exception as exc:
        return f"[error] {exc}"


def extract_bash_blocks(text: str) -> list[str]:
    return BASH_RE.findall(text)


def strip_bash_blocks(text: str) -> str:
    """Return the assistant text with bash blocks removed."""
    return BASH_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run(user_prompt: str) -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    print(_c(ANSI_CYAN, f"\n[harness] prompt: {user_prompt}\n"))

    for iteration in range(1, MAX_ITERATIONS + 1):
        # ---- call the model ------------------------------------------------
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        assistant_text: str = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        bash_blocks = extract_bash_blocks(assistant_text)
        thinking_text = strip_bash_blocks(assistant_text)

        # Print any reasoning/text the model produced
        if thinking_text:
            print(_c(ANSI_DIM, thinking_text))

        # ---- no bash blocks → final answer ----------------------------------
        if not bash_blocks:
            print(_c(ANSI_GREEN, "\n[harness] done.\n"))
            break

        # ---- execute each bash block ----------------------------------------
        result_parts: list[str] = []
        for cmd in bash_blocks:
            print(_c(ANSI_YELLOW, f"\n$ {cmd}"))
            output = run_bash(cmd)
            print(_c(ANSI_DIM, output))
            result_parts.append(
                f"<bash_result>\n$ {cmd}\n{output}\n</bash_result>"
            )

        # Feed all results back as a single user turn
        messages.append({"role": "user", "content": "\n".join(result_parts)})

    else:
        print(_c(ANSI_RED, f"\n[harness] reached max iterations ({MAX_ITERATIONS}).\n"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not sys.stdin.isatty():
        # support: echo "prompt" | python main.py
        prompt = sys.stdin.read().strip()
    elif len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("Usage:")
        print("  python main.py 'your prompt here'")
        print("  echo 'your prompt' | python main.py")
        sys.exit(1)

    if not prompt:
        print("Error: empty prompt.")
        sys.exit(1)

    run(prompt)


if __name__ == "__main__":
    main()
