use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::Command;
use std::time::Duration;

use anyhow::Result;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
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

fn candidate_hosts() -> Vec<String> {
    let mut hosts: Vec<String> = Vec::new();

    if let Ok(contents) = std::fs::read_to_string("/etc/resolv.conf") {
        for line in contents.lines() {
            if let Some(ip) = line.strip_prefix("nameserver ") {
                hosts.push(ip.trim().to_string());
            }
        }
    }

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

    hosts.push("localhost".to_string());
    hosts
}

async fn lm_studio_url(client: &Client) -> String {
    if let Ok(url) = std::env::var("LM_STUDIO_URL") {
        let base = url.trim_end_matches('/').to_string();
        return format!("{base}/v1/chat/completions");
    }

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

    eprintln!("{}", "\x1b[33m[harness] warning: could not reach LM Studio on any candidate host.\x1b[0m");
    eprintln!("{}", "\x1b[33m          Set LM_STUDIO_URL=http://<host>:1234 to override.\x1b[0m");
    format!("http://localhost:{}/v1/chat/completions", DEFAULT_PORT)
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are a helpful AI assistant with access to three tools: bash, search, and fetch.

To use a tool, output EXACTLY one of these XML tags — nothing else on those lines:

Run a shell command:
<bash>
command here
</bash>

Search the web:
<search>
search query here
</search>

Fetch a web page:
<fetch>
https://example.com
</fetch>

CRITICAL RULES — you MUST follow these exactly:
- Use ONLY the XML tag format shown above. No other format is allowed.
- Do NOT write <|tool_call>, ```bash, or any other syntax. Only <bash>, <search>, <fetch>.
- After a tool block, wait for the result — do not continue the response.
- Results come back inside <bash_result>, <search_result>, or <fetch_result> tags.
- When your task is fully done, reply with plain text and NO tool tags.

Example of correct usage:
User: what is the weather in Paris?
Assistant: <search>
weather in Paris today
</search>
<search_result>
...results...
</search_result>
The weather in Paris is currently 18°C and cloudy.
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
// Tool call extraction
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ToolCall {
    Bash(String),
    Search(String),
    Fetch(String),
}

fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let re = Regex::new(r"(?s)<(bash|search|fetch)>\s*(.*?)\s*</\1>").unwrap();
    re.captures_iter(text)
        .map(|cap| {
            let content = cap[2].to_string();
            match &cap[1] {
                "bash"   => ToolCall::Bash(content),
                "search" => ToolCall::Search(content),
                "fetch"  => ToolCall::Fetch(content),
                _        => unreachable!(),
            }
        })
        .collect()
}

fn strip_tool_calls(text: &str) -> String {
    let re = Regex::new(r"(?s)<(bash|search|fetch)>\s*.*?\s*</\1>").unwrap();
    re.replace_all(text, "").trim().to_string()
}

// ---------------------------------------------------------------------------
// Bash
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
                if !result.is_empty() { result.push('\n'); }
                result.push_str("[stderr]\n");
                result.push_str(stderr.trim_end());
            }
            if result.is_empty() { "(no output)".into() } else { result }
        }
        Err(e) => format!("[error] {e}"),
    }
}

// ---------------------------------------------------------------------------
// Web search (DuckDuckGo, no API key)
// ---------------------------------------------------------------------------

async fn run_search(client: &Client, query: &str) -> String {
    let resp = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
        .timeout(Duration::from_secs(10))
        .send()
        .await;

    let html = match resp {
        Ok(r) => match r.text().await {
            Ok(t) => t,
            Err(e) => return format!("[search error] {e}"),
        },
        Err(e) => return format!("[search error] {e}"),
    };

    let document = Html::parse_document(&html);
    let title_sel   = Selector::parse("a.result__a").unwrap();
    let snippet_sel = Selector::parse(".result__snippet").unwrap();

    let titles:   Vec<String> = document.select(&title_sel)
        .map(|e| e.text().collect::<String>().trim().to_string())
        .collect();
    let snippets: Vec<String> = document.select(&snippet_sel)
        .map(|e| e.text().collect::<String>().trim().to_string())
        .collect();

    let results: Vec<String> = titles.iter().zip(snippets.iter())
        .take(5)
        .enumerate()
        .map(|(i, (title, snippet))| format!("{}. {}\n   {}", i + 1, title, snippet))
        .collect();

    if results.is_empty() {
        "(no results)".to_string()
    } else {
        results.join("\n\n")
    }
}

// ---------------------------------------------------------------------------
// Web fetch (returns readable page text)
// ---------------------------------------------------------------------------

async fn run_fetch(client: &Client, url: &str) -> String {
    let url = url.trim();
    let resp = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
        .timeout(Duration::from_secs(15))
        .send()
        .await;

    let html = match resp {
        Ok(r) => match r.text().await {
            Ok(t) => t,
            Err(e) => return format!("[fetch error] {e}"),
        },
        Err(e) => return format!("[fetch error] {e}"),
    };

    // Strip tags, collapse whitespace, truncate to keep context reasonable
    let document = Html::parse_document(&html);
    let body_sel = Selector::parse("body").unwrap();
    let text: String = document
        .select(&body_sel)
        .next()
        .map(|b| b.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();

    // Collapse runs of whitespace
    let re = Regex::new(r"\s{2,}").unwrap();
    let clean = re.replace_all(text.trim(), " ").to_string();

    // Cap at ~4000 chars so it fits in context
    if clean.len() > 4000 {
        format!("{}… [truncated]", &clean[..4000])
    } else if clean.is_empty() {
        "(empty page)".to_string()
    } else {
        clean
    }
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

fn is_tty() -> bool { io::stdout().is_terminal() }

fn cyan(s: &str)   -> String { if is_tty() { format!("\x1b[36m{s}\x1b[0m") } else { s.into() } }
fn green(s: &str)  -> String { if is_tty() { format!("\x1b[32m{s}\x1b[0m") } else { s.into() } }
fn yellow(s: &str) -> String { if is_tty() { format!("\x1b[33m{s}\x1b[0m") } else { s.into() } }
fn red(s: &str)    -> String { if is_tty() { format!("\x1b[31m{s}\x1b[0m") } else { s.into() } }
fn dim(s: &str)    -> String { if is_tty() { format!("\x1b[2m{s}\x1b[0m")  } else { s.into() } }
fn bold(s: &str)   -> String { if is_tty() { format!("\x1b[1m{s}\x1b[0m")  } else { s.into() } }
fn magenta(s: &str)-> String { if is_tty() { format!("\x1b[35m{s}\x1b[0m") } else { s.into() } }

// ---------------------------------------------------------------------------
// Agent turn
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

        let tool_calls = extract_tool_calls(&assistant_text);
        let thought    = strip_tool_calls(&assistant_text);

        if !thought.is_empty() {
            println!("{}", dim(&thought));
        }

        if tool_calls.is_empty() {
            break;
        }

        let mut result_parts: Vec<String> = Vec::new();

        for call in &tool_calls {
            match call {
                ToolCall::Bash(cmd) => {
                    println!("{}", yellow(&format!("\n$ {cmd}")));
                    let output = run_bash(cmd);
                    println!("{}", dim(&output));
                    result_parts.push(format!("<bash_result>\n$ {cmd}\n{output}\n</bash_result>"));
                }
                ToolCall::Search(query) => {
                    println!("{}", magenta(&format!("\n[search] {query}")));
                    let output = run_search(client, query).await;
                    println!("{}", dim(&output));
                    result_parts.push(format!("<search_result>\n{output}\n</search_result>"));
                }
                ToolCall::Fetch(url) => {
                    println!("{}", cyan(&format!("\n[fetch] {url}")));
                    let output = run_fetch(client, url).await;
                    println!("{}", dim(&output));
                    result_parts.push(format!("<fetch_result>\n{output}\n</fetch_result>"));
                }
            }
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
// Interactive REPL
// ---------------------------------------------------------------------------

async fn interactive(client: &Client, url: &str) -> Result<()> {
    println!("{}", bold("Mini Agentic Harness — interactive mode"));
    println!("{}", dim("Tools: <bash>, <search>, <fetch>  |  Type 'exit' or Ctrl-D to quit.\n"));

    let mut messages = vec![
        Message { role: "system".into(), content: SYSTEM_PROMPT.into() },
    ];

    let stdin = io::stdin();
    loop {
        print!("{} ", cyan("you>"));
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => { println!("\n{}", dim("[bye]")); break; }
            Ok(_) => {}
            Err(e) => return Err(e.into()),
        }

        let input = line.trim().to_string();
        if input.is_empty() { continue; }
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
// Single-shot
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
        [] => interactive(&client, &url).await,
        _  => single_shot(&client, &url, args.join(" ")).await,
    }
}
