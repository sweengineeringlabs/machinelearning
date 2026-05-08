//! `llmc load` — HTTP load test an endpoint.
//!
//! Native implementation (no external oha binary). Two modes:
//!
//! - **Closed-loop** (default): C workers each fire as fast as they can.
//!   Good for "how fast can this server go?" measurements. Not CO-correct
//!   when you want tail latency at a target rate.
//! - **Open-loop / rate-targeted** (`--rate R`): a scheduler fires a request
//!   every 1/R seconds. Latency is measured from each request's scheduled
//!   time, not its actual send time, so a server stall inflates the
//!   percentiles honestly. This compensates for coordinated omission.
//!   Use this for SLO-grade tail measurements.
//!
//! Both modes record latencies into an `hdrhistogram::Histogram` and report
//! p50/p90/p95/p99/p99.9/max. Text or JSON output.
//!
//! Scope: fixed count (-n) or duration (-z); concurrency (-c); method (-X);
//! repeatable headers (-H); body inline or `@file` (-d); per-request timeout
//! (-t); rate target (-r); JSON output (--json).
//!
//! Not included (YAGNI): HTTP/3, SSE parsing, gRPC, response-body capture,
//! multi-URL randomization. Add when a real scenario needs them.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use hdrhistogram::Histogram;
use reqwest::{Client, Method};
use serde::Serialize;
use tokio::sync::{Mutex, Semaphore};

#[derive(Args, Debug)]
pub struct LoadArgs {
    /// Target URL (positional, e.g. http://127.0.0.1:8080/v1/chat/completions).
    pub url: String,

    /// Total number of requests to send. Mutually exclusive with --duration.
    #[arg(short = 'n', long, conflicts_with = "duration")]
    pub count: Option<u64>,

    /// Duration to run. Accepts 60s, 2m, 1h. Mutually exclusive with -n.
    #[arg(short = 'z', long)]
    pub duration: Option<String>,

    /// Number of concurrent requests in flight.
    #[arg(short = 'c', long, default_value_t = 4)]
    pub concurrency: usize,

    /// HTTP method.
    #[arg(short = 'X', long, default_value = "GET")]
    pub method: String,

    /// Header in the form Key:Value. Repeatable.
    #[arg(short = 'H', long = "header")]
    pub headers: Vec<String>,

    /// Request body. Use `@path/to/file` to load from a file.
    #[arg(short = 'd', long)]
    pub body: Option<String>,

    /// Per-request timeout in seconds.
    #[arg(short = 't', long, default_value_t = 30)]
    pub timeout_secs: u64,

    /// Target rate in requests per second. When set, switches to open-loop
    /// (scheduled) mode: a request is dispatched every 1/rate seconds and
    /// latency is measured from the scheduled time. This is the mode for
    /// SLO-grade tail measurements — it compensates for coordinated
    /// omission by keeping the send schedule even when the server stalls.
    /// Without --rate, runs in closed-loop mode (c workers firing as fast
    /// as possible).
    #[arg(short = 'r', long)]
    pub rate: Option<u64>,

    /// Emit a JSON result blob to stdout instead of a text summary.
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: LoadArgs) -> Result<()> {
    if args.count.is_none() && args.duration.is_none() {
        bail!("specify either -n <count> or -z <duration>");
    }
    if args.concurrency == 0 {
        bail!("--concurrency must be at least 1");
    }
    if let Some(r) = args.rate {
        if r == 0 {
            bail!("--rate must be at least 1");
        }
    }

    let runtime = tokio::runtime::Runtime::new().context("create tokio runtime")?;
    let report = runtime.block_on(execute(args))?;
    if let Some(out) = report.json_output.as_ref() {
        println!("{}", out);
    } else {
        print_summary(&report);
    }
    Ok(())
}

/// Accumulates per-request outcomes. Shared across workers behind a mutex.
/// A mutex on a histogram is fine here — recording a sample is a few ops,
/// and the workers are HTTP-bound, not contention-bound.
struct Accumulator {
    hist: Histogram<u64>,
    status_counts: std::collections::BTreeMap<u16, u64>,
    errors: u64,
    error_samples: Vec<String>,
}

impl Accumulator {
    fn new() -> Self {
        // 1µs precision, tracking up to ~60s (60_000_000µs). 3 significant digits.
        Self {
            hist: Histogram::<u64>::new_with_bounds(1, 60_000_000, 3)
                .expect("histogram config"),
            status_counts: Default::default(),
            errors: 0,
            error_samples: Vec::new(),
        }
    }

    fn record_success(&mut self, micros: u64, status: u16) {
        let _ = self.hist.record(micros);
        *self.status_counts.entry(status).or_insert(0) += 1;
    }

    fn record_error(&mut self, msg: String) {
        self.errors += 1;
        if self.error_samples.len() < 5 {
            self.error_samples.push(msg);
        }
    }
}

struct Report {
    json_output: Option<String>,
    total_requests: u64,
    successful: u64,
    errors: u64,
    wall_clock_secs: f64,
    rps: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    p99_9_ms: f64,
    max_ms: f64,
    min_ms: f64,
    status_counts: std::collections::BTreeMap<u16, u64>,
    error_samples: Vec<String>,
}

/// Shared request-dispatch context. Cheap to clone (all fields are either
/// `Arc`-cheap or small stack copies).
#[derive(Clone)]
struct DispatchCtx {
    client: Client,
    url: String,
    method: Method,
    headers: reqwest::header::HeaderMap,
    body: Option<Vec<u8>>,
}

/// Send one request, measure latency from `scheduled_at`, record into `acc`.
async fn fire_one(ctx: &DispatchCtx, scheduled_at: Instant, acc: &Mutex<Accumulator>) {
    let mut req = ctx
        .client
        .request(ctx.method.clone(), &ctx.url)
        .headers(ctx.headers.clone());
    if let Some(b) = &ctx.body {
        req = req.body(b.clone());
    }
    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let _ = resp.bytes().await; // drain so keep-alive works
            let micros = scheduled_at.elapsed().as_micros() as u64;
            acc.lock().await.record_success(micros, status);
        }
        Err(e) => {
            acc.lock().await.record_error(e.to_string());
        }
    }
}

async fn execute(args: LoadArgs) -> Result<Report> {
    let method = args
        .method
        .parse::<Method>()
        .with_context(|| format!("invalid HTTP method: {}", args.method))?;

    let body_bytes = match &args.body {
        Some(s) if s.starts_with('@') => {
            let path = PathBuf::from(&s[1..]);
            Some(fs::read(&path).with_context(|| format!("read body file {}", path.display()))?)
        }
        Some(s) => Some(s.as_bytes().to_vec()),
        None => None,
    };

    let mut header_map = reqwest::header::HeaderMap::new();
    for h in &args.headers {
        let (k, v) = h
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid header '{}', expected Key:Value", h))?;
        header_map.insert(
            k.trim().parse::<reqwest::header::HeaderName>()
                .with_context(|| format!("invalid header name '{}'", k))?,
            v.trim().parse::<reqwest::header::HeaderValue>()
                .with_context(|| format!("invalid header value for '{}'", k))?,
        );
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()
        .context("build HTTP client")?;

    let duration = if let Some(s) = &args.duration {
        Some(parse_duration(s)?)
    } else {
        None
    };

    let ctx = DispatchCtx {
        client,
        url: args.url.clone(),
        method,
        headers: header_map,
        body: body_bytes,
    };

    let acc = Arc::new(Mutex::new(Accumulator::new()));
    let deadline = duration.map(|d| Instant::now() + d);
    let started = Instant::now();

    if let Some(rate) = args.rate {
        run_open_loop(&ctx, &acc, started, deadline, args.count, rate, args.concurrency)
            .await;
    } else {
        run_closed_loop(&ctx, &acc, deadline, args.count, args.concurrency).await;
    }

    let elapsed = started.elapsed();
    let acc = acc.lock().await;

    let successful: u64 = acc.status_counts.values().sum();
    let total = successful + acc.errors;
    let wall = elapsed.as_secs_f64();
    let rps = if wall > 0.0 { total as f64 / wall } else { 0.0 };

    let to_ms = |v: u64| v as f64 / 1000.0;
    let report = Report {
        json_output: None,
        total_requests: total,
        successful,
        errors: acc.errors,
        wall_clock_secs: wall,
        rps,
        p50_ms: to_ms(acc.hist.value_at_quantile(0.50)),
        p90_ms: to_ms(acc.hist.value_at_quantile(0.90)),
        p95_ms: to_ms(acc.hist.value_at_quantile(0.95)),
        p99_ms: to_ms(acc.hist.value_at_quantile(0.99)),
        p99_9_ms: to_ms(acc.hist.value_at_quantile(0.999)),
        max_ms: to_ms(acc.hist.max()),
        min_ms: if successful > 0 { to_ms(acc.hist.min()) } else { 0.0 },
        status_counts: acc.status_counts.clone(),
        error_samples: acc.error_samples.clone(),
    };

    if args.json {
        let json = build_json(&report)?;
        Ok(Report {
            json_output: Some(json),
            ..report
        })
    } else {
        Ok(report)
    }
}

/// Closed-loop: `concurrency` workers, each grabs a request from the
/// shared counter and fires it. Next request starts only when the previous
/// finishes. Good for "max throughput" / stress-testing. Latency is
/// measured from actual-send, so it does NOT compensate for coordinated
/// omission.
async fn run_closed_loop(
    ctx: &DispatchCtx,
    acc: &Arc<Mutex<Accumulator>>,
    deadline: Option<Instant>,
    count: Option<u64>,
    concurrency: usize,
) {
    let remaining = Arc::new(AtomicU64::new(count.unwrap_or(u64::MAX)));
    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let ctx = ctx.clone();
        let acc = Arc::clone(acc);
        let remaining = Arc::clone(&remaining);
        handles.push(tokio::spawn(async move {
            loop {
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        break;
                    }
                }
                let prev = remaining.fetch_sub(1, Ordering::Relaxed);
                if prev == 0 {
                    remaining.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                let t0 = Instant::now();
                fire_one(&ctx, t0, &acc).await;
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }
}

/// Open-loop / rate-targeted: a scheduler dispatches a request every
/// `1/rate` seconds. Latency is measured from the *scheduled* time, not
/// the actual send time. When the server stalls, new requests still get
/// scheduled on time; they wait on the concurrency semaphore and their
/// measured latency includes the wait. This is the correct way to
/// measure tail latency against an SLO — it compensates for coordinated
/// omission by refusing to "let the client politely wait" during a stall.
///
/// `concurrency` caps the number of requests in flight simultaneously.
/// If it's exhausted, the scheduler blocks on the semaphore, which
/// itself is CO-correct: the next slot gets served when one frees up.
async fn run_open_loop(
    ctx: &DispatchCtx,
    acc: &Arc<Mutex<Accumulator>>,
    started: Instant,
    deadline: Option<Instant>,
    count: Option<u64>,
    rate: u64,
    concurrency: usize,
) {
    let interval_nanos = 1_000_000_000u64 / rate;
    let interval = Duration::from_nanos(interval_nanos);
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let total = count.unwrap_or(u64::MAX);

    let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
    for i in 0..total {
        // Scheduled send time for request i relative to start.
        let scheduled_at = started + interval * (i as u32);

        if let Some(d) = deadline {
            if scheduled_at >= d {
                break;
            }
        }

        // Sleep until the scheduled tick, unless we're already late.
        let now = Instant::now();
        if scheduled_at > now {
            tokio::time::sleep(scheduled_at - now).await;
        }

        // Acquire a concurrency slot. If the server is stalled and the
        // slot isn't free yet, this await is exactly the CO-correct
        // "the client is waiting" latency contribution.
        let permit = match semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break, // semaphore closed
        };

        let ctx = ctx.clone();
        let acc = Arc::clone(acc);
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            fire_one(&ctx, scheduled_at, &acc).await;
        }));
    }
    for h in handles {
        let _ = h.await;
    }
}

#[derive(Serialize)]
struct JsonReport<'a> {
    total_requests: u64,
    successful: u64,
    errors: u64,
    wall_clock_secs: f64,
    rps: f64,
    latency_ms: JsonLatency,
    status_counts: &'a std::collections::BTreeMap<u16, u64>,
    error_samples: &'a [String],
}

#[derive(Serialize)]
struct JsonLatency {
    min: f64,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    p99_9: f64,
    max: f64,
}

fn build_json(r: &Report) -> Result<String> {
    let jr = JsonReport {
        total_requests: r.total_requests,
        successful: r.successful,
        errors: r.errors,
        wall_clock_secs: r.wall_clock_secs,
        rps: r.rps,
        latency_ms: JsonLatency {
            min: r.min_ms,
            p50: r.p50_ms,
            p90: r.p90_ms,
            p95: r.p95_ms,
            p99: r.p99_ms,
            p99_9: r.p99_9_ms,
            max: r.max_ms,
        },
        status_counts: &r.status_counts,
        error_samples: &r.error_samples,
    };
    serde_json::to_string_pretty(&jr).context("serialize JSON report")
}

fn print_summary(r: &Report) {
    println!();
    println!("=== RESULTS ===");
    println!("Total requests:  {}", r.total_requests);
    println!("Successful:      {}", r.successful);
    println!("Errors:          {}", r.errors);
    println!("Wall clock:      {:.2}s", r.wall_clock_secs);
    println!("Requests/sec:    {:.2}", r.rps);
    println!();
    println!("=== LATENCY (ms) ===");
    println!("  min:   {:>8.2}", r.min_ms);
    println!("  p50:   {:>8.2}", r.p50_ms);
    println!("  p90:   {:>8.2}", r.p90_ms);
    println!("  p95:   {:>8.2}", r.p95_ms);
    println!("  p99:   {:>8.2}", r.p99_ms);
    println!("  p99.9: {:>8.2}", r.p99_9_ms);
    println!("  max:   {:>8.2}", r.max_ms);
    println!();
    println!("=== STATUS CODES ===");
    for (code, count) in &r.status_counts {
        println!("  {}: {}", code, count);
    }
    if !r.error_samples.is_empty() {
        println!();
        println!("=== ERROR SAMPLES (first {}) ===", r.error_samples.len());
        for e in &r.error_samples {
            println!("  {}", e);
        }
    }
}

fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    let (num_part, unit) = s
        .trim_end_matches(|c: char| c.is_alphabetic())
        .parse::<u64>()
        .ok()
        .map(|n| (n, &s[s.len() - (s.len() - n.to_string().len())..]))
        .ok_or_else(|| anyhow::anyhow!("invalid duration '{}', expected e.g. 60s, 2m, 1h", s))?;
    let (num, unit) = (num_part, unit.trim());
    let secs = match unit {
        "" | "s" => num,
        "m" => num * 60,
        "h" => num * 3600,
        other => bail!("unknown duration unit '{}', expected s, m, or h", other),
    };
    Ok(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_seconds_default_no_unit() {
        assert_eq!(parse_duration("60").unwrap(), Duration::from_secs(60));
    }

    #[test]
    fn test_parse_duration_seconds_with_s() {
        assert_eq!(parse_duration("45s").unwrap(), Duration::from_secs(45));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("3m").unwrap(), Duration::from_secs(180));
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
    }

    #[test]
    fn test_parse_duration_unknown_unit_rejected() {
        assert!(parse_duration("5d").is_err());
    }
}
