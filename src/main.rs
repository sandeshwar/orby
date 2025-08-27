use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use eframe::egui;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use tokio::runtime::Builder as TokioRtBuilder;

mod llm;
use llm::{ChatMessage, LlmClient, LlmConfig, Role};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([500.0, 460.0])
            .with_title("Voice Control"),
        ..Default::default()
    };

    eframe::run_native(
        "Voice Control",
        options,
        Box::new(|_cc| Box::new(App::default())),
    )
}

impl App {
    fn start_llm_request(&mut self) {
        if self.llm_busy {
            return;
        }
        // Reset output
        self.llm_response.clear();
        if let Ok(mut q) = self.pending_llm_deltas.lock() { q.clear(); }
        self.pending_llm_done.store(false, Ordering::Relaxed);
        self.llm_busy = true;
        self.status = if self.use_streaming { "LLM (streaming)…".to_string() } else { "LLM…".to_string() };

        let transcript = self.transcript.clone();
        let streaming = self.use_streaming;
        let deltas = self.pending_llm_deltas.clone();
        let done = self.pending_llm_done.clone();

        thread::spawn(move || {
            // Build client/config from env each request to remain provider-agnostic
            // Clone for runtime creation error branch
            let deltas_err = deltas.clone();
            let done_err = done.clone();

            let run = || async move {
                let cfg = match LlmConfig::from_env() {
                    Ok(c) => c,
                    Err(e) => {
                        if let Ok(mut q) = deltas.lock() { q.push(format!("[error] {e}")); }
                        done.store(true, Ordering::Relaxed);
                        return;
                    }
                };
                let client = match LlmClient::new(cfg) {
                    Ok(c) => c,
                    Err(e) => {
                        if let Ok(mut q) = deltas.lock() { q.push(format!("[error] {e}")); }
                        done.store(true, Ordering::Relaxed);
                        return;
                    }
                };

                let messages = vec![ChatMessage { role: Role::User, content: transcript }];

                if streaming {
                    let _ = client.send_stream(&messages, |delta| {
                        if let Ok(mut q) = deltas.lock() { q.push(delta.to_string()); }
                    }).await.map_err(|e| {
                        if let Ok(mut q) = deltas.lock() { q.push(format!("[error] {e}")); }
                    });
                } else {
                    match client.send(&messages).await {
                        Ok(full) => { if let Ok(mut q) = deltas.lock() { q.push(full); } }
                        Err(e) => { if let Ok(mut q) = deltas.lock() { q.push(format!("[error] {e}")); } }
                    }
                }
                done.store(true, Ordering::Relaxed);
            };

            // Run the async block on a fresh multi-threaded runtime
            let rt = TokioRtBuilder::new_multi_thread().enable_all().build();
            match rt {
                Ok(rt) => { rt.block_on(run()); }
                Err(e) => {
                    if let Ok(mut q) = deltas_err.lock() { q.push(format!("[error] tokio rt: {e}")); }
                    done_err.store(true, Ordering::Relaxed);
                }
            }
        });
    }
}

#[derive(Default)]
struct App {
    listening: bool,
    ptt_active: bool,
    status: String,
    stream: Option<Stream>,
    // simple audio level meter (0.0..1.0) updated by the input callback
    level: Arc<Mutex<f32>>,
    // push-to-talk shared flag and buffer for captured audio (mono f32 @16k)
    ptt_flag: Arc<AtomicBool>,
    audio_buf: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    // transcription state
    transcribing: bool,
    transcript: String,
    pending_transcript: Arc<Mutex<Option<String>>>,
    // LLM state
    use_streaming: bool,
    llm_busy: bool,
    llm_response: String,
    pending_llm_deltas: Arc<Mutex<Vec<String>>>,
    pending_llm_done: Arc<AtomicBool>,
    // optional cached client (rebuilt per request if env changes)
    llm_client: Option<LlmClient>,
}

impl App {
    fn start_listening(&mut self) {
        if self.listening {
            return;
        }

        // clear any previous audio
        if let Ok(mut b) = self.audio_buf.lock() { b.clear(); }

        match build_input_stream(self.level.clone(), self.ptt_flag.clone(), self.audio_buf.clone()) {
            Ok((stream, sr)) => {
                if let Err(e) = stream.play() {
                    self.status = format!("Error starting stream: {e}");
                    return;
                }
                self.stream = Some(stream);
                self.listening = true;
                self.sample_rate = sr;
                self.status = "Listening…".to_string();
            }
            Err(e) => {
                self.status = format!("Failed to init input: {e}");
            }
        }
    }

    fn stop_listening(&mut self) {
        if let Some(stream) = &self.stream {
            let _ = stream.pause();
        }
        self.stream = None;
        self.listening = false;
        self.status = "Ready".to_string();
        if let Ok(mut lvl) = self.level.lock() {
            *lvl = 0.0;
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle PTT chord: UP + RIGHT
        let chord_down = ctx.input(|i| i.key_down(egui::Key::ArrowUp))
            && ctx.input(|i| i.key_down(egui::Key::ArrowRight));

        if chord_down && !self.ptt_active && self.listening {
            self.ptt_active = true;
            self.status = "Listening…".to_string();
            self.ptt_flag.store(true, Ordering::Relaxed);
            if let Ok(mut b) = self.audio_buf.lock() { b.clear(); }
        }
        if !chord_down && self.ptt_active {
            self.ptt_active = false;
            self.ptt_flag.store(false, Ordering::Relaxed);
            // Kick off transcription if we have audio and not already transcribing
            if !self.transcribing {
                if let Ok(buf) = self.audio_buf.lock() {
                    if !buf.is_empty() {
                        let pcm = buf.clone();
                        drop(buf);
                        self.transcribing = true;
                        self.status = "Transcribing…".to_string();
                        let pending = self.pending_transcript.clone();
                        thread::spawn(move || {
                            let res = transcribe_whisper(pcm);
                            if let Ok(mut p) = pending.lock() {
                                *p = Some(match res { Ok(s) => s, Err(e) => format!("[error] {e}") });
                            }
                        });
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Voice Control System");
            ui.add_space(8.0);

            // Status row with optional spinner
            ui.horizontal(|ui| {
                if self.ptt_active || self.transcribing || self.llm_busy {
                    ui.add(egui::widgets::Spinner::new());
                }
                ui.label(if self.status.is_empty() { "Ready" } else { &self.status });
            });

            ui.add_space(8.0);

            // Simple level meter
            if let Ok(level) = self.level.lock() {
                let frac = (*level).clamp(0.0, 1.0);
                ui.add(egui::ProgressBar::new(frac).text("Input level"));
            }

            ui.add_space(12.0);

            // Toggle button
            let btn_label = if self.listening { "Stop Listening" } else { "Start Listening" };
            if ui.button(btn_label).clicked() {
                if self.listening {
                    self.stop_listening();
                } else {
                    self.start_listening();
                }
            }

            ui.add_space(12.0);
            ui.small("PTT: UP+RIGHT");

            ui.add_space(8.0);
            if !self.transcript.is_empty() {
                ui.label("Transcript:");
                ui.separator();
                ui.label(&self.transcript);
            }

            ui.add_space(12.0);
            ui.checkbox(&mut self.use_streaming, "Stream LLM reply");

            ui.add_space(8.0);
            if !self.llm_response.is_empty() || self.llm_busy {
                ui.label("LLM Response:");
                ui.separator();
                ui.label(&self.llm_response);
            }
        });

        // Drain pending transcript from worker thread
        if self.transcribing {
            // Take the transcript out while holding the lock, then drop the guard
            let mut taken: Option<String> = None;
            if let Ok(mut p) = self.pending_transcript.lock() {
                taken = p.take();
            }
            if let Some(text) = taken {
                self.transcribing = false;
                self.transcript = text;
                self.status = "Ready".to_string();
                // Automatically send to LLM when a new transcript arrives
                if !self.transcript.is_empty() {
                    self.start_llm_request();
                }
            }
        }

        // Drain streaming LLM deltas
        if self.llm_busy {
            if let Ok(mut q) = self.pending_llm_deltas.lock() {
                if !q.is_empty() {
                    for chunk in q.drain(..) {
                        self.llm_response.push_str(&chunk);
                    }
                }
            }
            if self.pending_llm_done.load(Ordering::Relaxed) {
                self.pending_llm_done.store(false, Ordering::Relaxed);
                self.llm_busy = false;
                self.status = "Ready".to_string();
            }
        }
    }
}

fn build_input_stream(
    level: Arc<Mutex<f32>>,
    ptt_flag: Arc<AtomicBool>,
    audio_buf: Arc<Mutex<Vec<f32>>>,
) -> Result<(Stream, u32), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "No default input device".to_string())?;

    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to get default input config: {e}"))?;

    let sample_format = config.sample_format();
    let cfg = config.config();
    let sr = cfg.sample_rate.0;

    let err_fn = |e| eprintln!("CPAL stream error: {e}");

    // Build a stream matching the device's native sample format
    let stream = match sample_format {
        SampleFormat::F32 => build_stream_f32(&device, &cfg, level, ptt_flag, audio_buf, err_fn),
        SampleFormat::I16 => build_stream_i16(&device, &cfg, level, ptt_flag, audio_buf, err_fn),
        SampleFormat::U16 => build_stream_u16(&device, &cfg, level, ptt_flag, audio_buf, err_fn),
        other => return Err(format!("Unsupported sample format: {other:?}")),
    }?;

    Ok((stream, sr))
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    level: Arc<Mutex<f32>>,
    ptt_flag: Arc<AtomicBool>,
    audio_buf: Arc<Mutex<Vec<f32>>>,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<Stream, String> {
    let channels = config.channels as usize;
    let src_rate = config.sample_rate.0 as f32;
    let step = (src_rate / 16_000.0).max(1.0);
    let mut acc = 0.0f32;

    let data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for frame in data.chunks(channels) {
            let s = frame[0];
            sum += s * s;
            count += 1;
            // push-to-talk buffering with naive decimation to ~16kHz
            if ptt_flag.load(Ordering::Relaxed) {
                acc += 1.0;
                if acc >= step {
                    if let Ok(mut b) = audio_buf.lock() { b.push(s); }
                    acc -= step;
                }
            }
        }
        if count > 0 {
            let rms = (sum / count as f32).sqrt();
            if let Ok(mut lvl) = level.lock() {
                let alpha = 0.2f32;
                *lvl = (*lvl) * (1.0 - alpha) + rms * alpha;
            }
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .map_err(|e| format!("Failed to build input stream: {e}"))
}

fn transcribe_whisper(pcm_mono_16k: Vec<f32>) -> Result<String, String> {
    let model = std::env::var("WHISPER_MODEL").unwrap_or_else(|_| "models/ggml-small.en.bin".to_string());
    let ctx = WhisperContext::new_with_params(&model, WhisperContextParameters::default())
        .map_err(|e| format!("load model: {e}"))?;
    let mut state = ctx.create_state().map_err(|e| format!("create state: {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(4) as i32);
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_single_segment(true);
    // run
    state.full(params, &pcm_mono_16k).map_err(|e| format!("full: {e}"))?;
    // collect segments
    let mut out = String::new();
    let n = state.full_n_segments().map_err(|e| format!("n_segments: {e}"))?;
    for i in 0..n {
        let seg = state.full_get_segment_text(i).map_err(|e| format!("segment_text: {e}"))?;
        out.push_str(&seg);
    }
    Ok(out.trim().to_string())
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    level: Arc<Mutex<f32>>,
    ptt_flag: Arc<AtomicBool>,
    audio_buf: Arc<Mutex<Vec<f32>>>,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<Stream, String> {
    let channels = config.channels as usize;
    let src_rate = config.sample_rate.0 as f32;
    let step = (src_rate / 16_000.0).max(1.0);
    let mut acc = 0.0f32;

    let data_fn = move |data: &[i16], _: &cpal::InputCallbackInfo| {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for frame in data.chunks(channels) {
            let s = frame[0] as f32 / i16::MAX as f32;
            sum += s * s;
            count += 1;
            if ptt_flag.load(Ordering::Relaxed) {
                acc += 1.0;
                if acc >= step {
                    if let Ok(mut b) = audio_buf.lock() { b.push(s); }
                    acc -= step;
                }
            }
        }
        if count > 0 {
            let rms = (sum / count as f32).sqrt();
            if let Ok(mut lvl) = level.lock() {
                let alpha = 0.2f32;
                *lvl = (*lvl) * (1.0 - alpha) + rms * alpha;
            }
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .map_err(|e| format!("Failed to build input stream: {e}"))
}

fn build_stream_u16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    level: Arc<Mutex<f32>>,
    ptt_flag: Arc<AtomicBool>,
    audio_buf: Arc<Mutex<Vec<f32>>>,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<Stream, String> {
    let channels = config.channels as usize;
    let src_rate = config.sample_rate.0 as f32;
    let step = (src_rate / 16_000.0).max(1.0);
    let mut acc = 0.0f32;

    let data_fn = move |data: &[u16], _: &cpal::InputCallbackInfo| {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for frame in data.chunks(channels) {
            // Map 0..u16::MAX to ~-1.0..1.0
            let s = (frame[0] as f32 / u16::MAX as f32) * 2.0 - 1.0;
            sum += s * s;
            count += 1;
            if ptt_flag.load(Ordering::Relaxed) {
                acc += 1.0;
                if acc >= step {
                    if let Ok(mut b) = audio_buf.lock() { b.push(s); }
                    acc -= step;
                }
            }
        }
        if count > 0 {
            let rms = (sum / count as f32).sqrt();
            if let Ok(mut lvl) = level.lock() {
                let alpha = 0.2f32;
                *lvl = (*lvl) * (1.0 - alpha) + rms * alpha;
            }
        }
    };

    device
        .build_input_stream(config, data_fn, err_fn, None)
        .map_err(|e| format!("Failed to build input stream: {e}"))
}
