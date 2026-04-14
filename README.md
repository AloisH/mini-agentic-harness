# mini-agentic-harness

A minimal agentic loop in Rust. Send a prompt to a local LLM (via [LM Studio](https://lmstudio.ai/)), let it run bash commands, and get a final answer — like a tiny Claude Code.

## How it works

```
you ──► harness ──► LM Studio (local LLM)
                        │
                        │  <bash>ls /tmp</bash>
                        ▼
                    sh -c "ls /tmp"
                        │
                        │  <bash_result>…</bash_result>
                        ▼
                    LM Studio (keeps reasoning)
                        │
                        │  (no more <bash> blocks)
                        ▼
                    final answer ──► you
```

The LLM can emit any number of `<bash>` blocks. Each is executed locally and the output is fed back. The loop repeats until the model replies with no bash blocks.

## Requirements

- Rust (stable) — [rustup.rs](https://rustup.rs)
- [LM Studio](https://lmstudio.ai/) running on port `1234` with a model loaded

## Build

```bash
cargo build --release
# binary at: ./target/release/mah
```

## Usage

### Interactive mode (chat session)

```bash
cargo run
# or after building:
./target/release/mah
```

Keeps the full conversation history across turns. Type `exit` or hit `Ctrl-D` to quit.

```
Mini Agentic Harness — interactive mode
Type your message and press Enter. Type 'exit' or Ctrl-D to quit.

you> what files are in /tmp?
you> now delete the oldest one
you> exit
```

### Single-shot mode

```bash
cargo run -- 'list the 5 biggest files in my home directory'
# or after building:
./target/release/mah 'list the 5 biggest files in my home directory'
```

### Piped input

```bash
echo 'what kernel am I on?' | cargo run
```

## LM Studio setup

1. Open LM Studio and load a model
2. Go to **Local Server** (the `<->` icon on the left sidebar)
3. Set the host to `0.0.0.0` (not `127.0.0.1`) — required to accept connections from WSL2
4. Click **Start Server**
5. Confirm it's reachable from WSL: `curl http://<host>:1234/v1/models`

### WSL2 host detection

The harness auto-detects the Windows host IP at startup by probing candidates
(`/etc/resolv.conf` nameserver, default gateway, localhost) and using the first
one that responds on port `1234`. The resolved URL is printed on every run:

```
[harness] LM Studio → http://169.254.83.107:1234/v1/chat/completions
```

If auto-detection picks the wrong address, override it with an env var:

```bash
export LM_STUDIO_URL=http://169.254.83.107:1234
cargo run
```

## Configuration

All tunables are constants at the top of `src/main.rs`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_PORT` | `1234` | LM Studio server port |
| `MAX_ITERATIONS` | `20` | Max bash→reply cycles per turn before giving up |

`LM_STUDIO_URL` env var skips host probing entirely and uses the given address.

## Bash tool syntax

The system prompt instructs the LLM to wrap shell commands like this:

```
<bash>
your command here
</bash>
```

Results are returned as:

```
<bash_result>
$ your command here
output...
</bash_result>
```

Works with any instruction-following model (Mistral, Llama 3, Qwen, Deepseek, etc.).
