use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::Command;
use std::time::Duration;

use anyhow::Result;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DEFAULT_PORT: u16 = 1234;
const MODEL: &str = "local-model"; // LM Studio ignores this
const MAX_ITERATIONS: usize = 20;

// ---------------------------------------------------------------------------
// Host detection — probe multiple candidates
// ---------------------------------------------------------------------------

/// Collect every plausible host IP for the Windows side of WSL2.
fn candidate_hosts() -> Vec<String> {
    let mut hosts: Vec<String> = Vec::new();

    // 1. /etc/resolv.conf nameserver
    if let Ok(contents) = std::fs::read_to_string("/etc/resolv.conf") {
        for line in contents.lines() {
            if let Some(ip) = line.strip_prefix("nameserver ") {
                hosts.push(ip.trim().to_string());
            }
        }
    }

    // 2. Default gateway from /proc/net/route (little-endian hex)
    if let Ok(contents) = std::fs::read_to_string("/proc/net/route") {
        for line in contents.lines().skip(1) {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 3 { continue; }
            let dest = u32::from_str_radix(cols[1], 16).unwrap_or(1);
            let gw   = u32::from_str_radix(cols[2], 16).unwrap_or(0);
            if dest == 0 && gw != 0 {
                let b = gw.to_le_bytes();
                hosts.push(format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3]));
            }
        }
    }

    // 3. localhost always last
    hosts.push("localhost".to_string());
    hosts
}

/// Returns the base URL for the LM Studio API.
///
/// Resolution order:
///   1. `LM_STUDIO_URL` env var  — full override, e.g. "http://192.168.1.5:1234"
///   2. Probe: nameserver, default gateway, localhost — first one that responds wins
async fn lm_studio_url(client: &Client) -> String {
    // 1. Explicit env override — skip probing entirely
    if let Ok(url) = std::env::var("LM_STUDIO_URL") {
        let base = url.trim_end_matches('/').to_string();
        return format!("{base}/v1/chat/completions");
    }

    // 2. Probe each candidate
    for host in candidate_hosts() {
        let probe = format!("http://{}:{}/v1/models", host, DEFAULT_PORT);
        let ok = client
            .get(&probe)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if ok {
            return format!("http://{}:{}/v1/chat/completions", host, DEFAULT_PORT);
        }
    }

    // 3. Nothing responded — fall back to localhost and let the user see the error
    eprintln!("{}", "\x1b[33m[harness] warning: could not reach LM Studio on any candidate host.\x1b[0m");
    eprintln!("{}", "\x1b[33m          Set LM_STUDIO_URL=http://<host>:1234 to override.\x1b[0m");
    format!("http://localhost:{}/v1/chat/completions", DEFAULT_PORT)
}

const SYSTEM_PROMPT: &str = "\
You are a helpful AI assistant that can run bash commands on the user's machine.

When you need to execute a shell command, wrap it like this:

<bash>
your command here
</bash>

Rules:
- You can use multiple <bash> blocks in one reply.
- After each block is executed you will receive the output inside <bash_result> tags.
- Keep reasoning and issuing bash blocks until the task is fully done.
- When you have a final answer, reply normally with NO <bash> blocks.
";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

// ---------------------------------------------------------------------------
// Bash execution
// ---------------------------------------------------------------------------

fn run_bash(command: &str) -> String {
    match Command::new("sh").arg("-c").arg(command).output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(stdout.trim_end());
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("[stderr]\n");
                result.push_str(stderr.trim_end());
            }
            if result.is_empty() { "(no output)".into() } else { result }
        }
        Err(e) => format!("[error] {e}"),
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn extract_bash_blocks(text: &str) -> Vec<String> {
    let re = Regex::new(r"(?s)<bash>\s*(.*?)\s*</bash>").unwrap();
    re.captures_iter(text).map(|cap| cap[1].to_string()).collect()
}

fn strip_bash_blocks(text: &str) -> String {
    let re = Regex::new(r"(?s)<bash>\s*.*?\s*</bash>").unwrap();
    re.replace_all(text, "").trim().to_string()
}

// ---------------------------------------------------------------------------
// ANSI helpers (only when stdout is a tty)
// ---------------------------------------------------------------------------

fn is_tty() -> bool {
    io::stdout().is_terminal()
}

fn cyan(s: &str)   -> String { if is_tty() { format!("\x1b[36m{s}\x1b[0m") } else { s.into() } }
fn green(s: &str)  -> String { if is_tty() { format!("\x1b[32m{s}\x1b[0m") } else { s.into() } }
fn yellow(s: &str) -> String { if is_tty() { format!("\x1b[33m{s}\x1b[0m") } else { s.into() } }
fn red(s: &str)    -> String { if is_tty() { format!("\x1b[31m{s}\x1b[0m") } else { s.into() } }
fn dim(s: &str)    -> String { if is_tty() { format!("\x1b[2m{s}\x1b[0m")  } else { s.into() } }
fn bold(s: &str)   -> String { if is_tty() { format!("\x1b[1m{s}\x1b[0m")  } else { s.into() } }

// ---------------------------------------------------------------------------
// Single agent turn — runs the LLM+bash loop for one user message.
// Appends everything (assistant replies + tool results) to `messages`.
// ---------------------------------------------------------------------------

async fn agent_turn(client: &Client, messages: &mut Vec<Message>, url: &str) -> Result<()> {
    for iteration in 1..=MAX_ITERATIONS {
        let body = json!({ "model": MODEL, "messages": messages });

        let raw = client
            .post(url)
            .header("Authorization", "Bearer lm-studio")
            .json(&body)
            .send()
            .await?
            .text()
            .await?;

        let resp: ChatResponse = serde_json::from_str(&raw).map_err(|e| {
            anyhow::anyhow!("LM Studio returned an unexpected response ({e}):\n{raw}")
        })?;

        let assistant_text = resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        messages.push(Message { role: "assistant".into(), content: assistant_text.clone() });

        let blocks  = extract_bash_blocks(&assistant_text);
        let thought = strip_bash_blocks(&assistant_text);

        if !thought.is_empty() {
            println!("{}", dim(&thought));
        }

        if blocks.is_empty() {
            // No bash blocks → the model is done for this turn.
            break;
        }

        let mut result_parts: Vec<String> = Vec::new();
        for cmd in &blocks {
            println!("{}", yellow(&format!("\n$ {cmd}")));
            let output = run_bash(cmd);
            println!("{}", dim(&output));
            result_parts.push(format!("<bash_result>\n$ {cmd}\n{output}\n</bash_result>"));
        }

        messages.push(Message {
            role: "user".into(),
            content: result_parts.join("\n"),
        });

        if iteration == MAX_ITERATIONS {
            eprintln!("{}", red(&format!("\n[harness] reached max iterations ({MAX_ITERATIONS}).")));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive REPL — keeps history across turns, type 'exit' or Ctrl-D to quit.
// ---------------------------------------------------------------------------

async fn interactive(client: &Client, url: &str) -> Result<()> {
    println!("{}", bold("Mini Agentic Harness — interactive mode"));
    println!("{}", dim("Type your message and press Enter. Type 'exit' or Ctrl-D to quit.\n"));

    let mut messages = vec![
        Message { role: "system".into(), content: SYSTEM_PROMPT.into() },
    ];

    let stdin = io::stdin();
    loop {
        // Print the prompt
        print!("{} ", cyan("you>"));
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // Ctrl-D / EOF
                println!("\n{}", dim("[bye]"));
                break;
            }
            Ok(_) => {}
            Err(e) => return Err(e.into()),
        }

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            println!("{}", dim("[bye]"));
            break;
        }

        messages.push(Message { role: "user".into(), content: input });
        println!();

        agent_turn(client, &mut messages, url).await?;
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single-shot mode — one prompt, one agentic turn, then exit.
// ---------------------------------------------------------------------------

async fn single_shot(client: &Client, url: &str, prompt: String) -> Result<()> {
    println!("{}\n", cyan(&format!("[harness] {prompt}")));

    let mut messages = vec![
        Message { role: "system".into(), content: SYSTEM_PROMPT.into() },
        Message { role: "user".into(),   content: prompt },
    ];

    agent_turn(client, &mut messages, url).await?;
    println!("{}", green("[harness] done."));
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let url = lm_studio_url(&client).await;
    println!("{}", dim(&format!("[harness] LM Studio → {url}")));

    // Piped input → single shot
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        let prompt = buf.trim().to_string();
        if prompt.is_empty() {
            eprintln!("Error: empty prompt.");
            std::process::exit(1);
        }
        return single_shot(&client, &url, prompt).await;
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.as_slice() {
        [] => {
            // No args + tty → interactive mode
            interactive(&client, &url).await
        }
        _ => {
            // Args provided → single shot
            single_shot(&client, &url, args.join(" ")).await
        }
    }
}
