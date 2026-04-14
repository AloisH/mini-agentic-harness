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

const BASE_URL: &str = "http://localhost:1234/v1/chat/completions";
const MODEL: &str = "local-model"; // LM Studio ignores this
const MAX_ITERATIONS: usize = 20;

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

async fn agent_turn(client: &Client, messages: &mut Vec<Message>) -> Result<()> {
    for iteration in 1..=MAX_ITERATIONS {
        let body = json!({ "model": MODEL, "messages": messages });

        let resp = client
            .post(BASE_URL)
            .header("Authorization", "Bearer lm-studio")
            .json(&body)
            .send()
            .await?
            .json::<ChatResponse>()
            .await?;

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

async fn interactive(client: &Client) -> Result<()> {
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

        agent_turn(client, &mut messages).await?;
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single-shot mode — one prompt, one agentic turn, then exit.
// ---------------------------------------------------------------------------

async fn single_shot(client: &Client, prompt: String) -> Result<()> {
    println!("{}\n", cyan(&format!("[harness] {prompt}")));

    let mut messages = vec![
        Message { role: "system".into(), content: SYSTEM_PROMPT.into() },
        Message { role: "user".into(),   content: prompt },
    ];

    agent_turn(client, &mut messages).await?;
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

    // Piped input → single shot
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        let prompt = buf.trim().to_string();
        if prompt.is_empty() {
            eprintln!("Error: empty prompt.");
            std::process::exit(1);
        }
        return single_shot(&client, prompt).await;
    }

    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.as_slice() {
        [] => {
            // No args + tty → interactive mode
            interactive(&client).await
        }
        _ => {
            // Args provided → single shot
            single_shot(&client, args.join(" ")).await
        }
    }
}
