use std::env;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

/// Provider-agnostic configuration. Defaults target an OpenAI-compatible API.
#[derive(Clone, Debug)]
pub struct LlmConfig {
    pub base_url: String,   // e.g., https://api.deepseek.com
    pub api_key: String,    // e.g., env LLM_API_KEY
    pub model: String,      // e.g., deepseek-chat
    pub timeout_secs: u64,  // request timeout
}

impl LlmConfig {
    pub fn from_env() -> Result<Self, String> {
        // Common envs supported
        // Prefer LLM_* first, then fall back to OPENAI_* naming
        let api_key = env::var("LLM_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .map_err(|_| "Missing LLM_API_KEY or OPENAI_API_KEY".to_string())?;
        let base_url = env::var("LLM_BASE_URL")
            .or_else(|_| env::var("OPENAI_API_BASE"))
            .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
        let model = env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());
        let timeout_secs = env::var("LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60);
        Ok(Self { base_url, api_key, model, timeout_secs })
    }
}

/// Simple chat message role
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, User, Assistant }

/// Chat message for OpenAI-compatible APIs
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Serialize, Debug)]
struct ChatCompletionsRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionsResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: Message,
}

#[derive(Deserialize, Debug)]
struct Message { content: String }

// Streaming (SSE) chunk structures for OpenAI-compatible APIs
#[derive(Deserialize, Debug)]
struct StreamResponseChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize, Debug)]
struct StreamChoice { delta: Delta, finish_reason: Option<String> }

#[derive(Deserialize, Debug)]
struct Delta { content: Option<String> }

/// Provider-agnostic client targeting OpenAI-compatible chat completions
#[derive(Clone)]
pub struct LlmClient {
    cfg: LlmConfig,
    http: reqwest::Client,
}

impl LlmClient {
    pub fn new(cfg: LlmConfig) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| format!("build http: {e}"))?;
        Ok(Self { cfg, http })
    }

    fn endpoint(&self) -> String {
        // Normalize base URL (allow user to provide with or without trailing slash)
        let mut base = self.cfg.base_url.trim_end_matches('/').to_string();
        base.push_str("/v1/chat/completions");
        base
    }

    /// Non-streaming request. Returns the first choice's content.
    pub async fn send(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let body = ChatCompletionsRequest { model: &self.cfg.model, messages, stream: None };
        let resp = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("send: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(format!("llm http {code}: {txt}"));
        }
        let data: ChatCompletionsResponse = resp.json().await.map_err(|e| format!("decode: {e}"))?;
        let content = data
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();
        Ok(content)
    }

    /// Streaming request. The `on_delta` callback is called for each content delta chunk.
    pub async fn send_stream<F>(&self, messages: &[ChatMessage], mut on_delta: F) -> Result<(), String>
    where
        F: FnMut(&str) + Send,
    {
        let body = ChatCompletionsRequest { model: &self.cfg.model, messages, stream: Some(true) };
        let mut resp = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("send: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(format!("llm http {code}: {txt}"));
        }
        // SSE comes as lines beginning with "data: ..."
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("stream: {e}"))?;
            buf.extend_from_slice(&chunk);
            // process by lines
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line = buf.drain(..=pos).collect::<Vec<u8>>();
                let line = String::from_utf8_lossy(&line);
                let l = line.trim();
                if l.is_empty() { continue; }
                if let Some(rest) = l.strip_prefix("data: ") {
                    if rest.trim() == "[DONE]" { return Ok(()); }
                    match serde_json::from_str::<StreamResponseChunk>(rest) {
                        Ok(ev) => {
                            if let Some(choice) = ev.choices.into_iter().next() {
                                if let Some(delta) = choice.delta.content {
                                    on_delta(&delta);
                                }
                            }
                        }
                        Err(_e) => { /* ignore parse errors on heartbeats */ }
                    }
                }
            }
        }
        Ok(())
    }
}
