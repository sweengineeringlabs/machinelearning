//! `llmc load` — HTTP load test an endpoint.
//!
//! Native implementation (no external oha binary). Fires N requests at C
//! concurrency, collects latency distribution via `hdrhistogram`, reports
//! percentiles + status code breakdown. JSON output mode for CI consumption.
//!
//! Scope: what we actually need for llmserv load testing.
//! - fixed total count (-n) OR fixed duration (-z)
//! - concurrency (-c) implemented via N concurrent tokio tasks draining a
//!   shared counter
//! - arbitrary method (-X), headers (-H), body (-d / -d @file)
//! - text summary or machine-readable JSON
//!
//! Not included (YAGNI): HTTP/3, SSE parsing, gRPC, rate targeting, response
//! body capture, multi-URL randomization. Add when a real scenario demands
//! it.

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
use tokio::sync::Mutex;

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

    let acc = Arc::new(Mutex::new(Accumulator::new()));
    let remaining = Arc::new(AtomicU64::new(args.count.unwrap_or(u64::MAX)));
    let deadline = duration.map(|d| Instant::now() + d);
    let started = Instant::now();

    let mut handles = Vec::with_capacity(args.concurrency);
    for _ in 0..args.concurrency {
        let client = client.clone();
        let url = args.url.clone();
        let method = method.clone();
        let headers = header_map.clone();
        let body = body_bytes.clone();
        let acc = Arc::clone(&acc);
        let remaining = Arc::clone(&remaining);
        let deadline = deadline;

        handles.push(tokio::spawn(async move {
            loop {
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        break;
                    }
                }
                // Decrement-and-test: bail when the budget runs out.
                let prev = remaining.fetch_sub(1, Ordering::Relaxed);
                if prev == 0 {
                    remaining.fetch_add(1, Ordering::Relaxed);
                    break;
                }

                let mut req = client.request(method.clone(), &url).headers(headers.clone());
                if let Some(b) = &body {
                    req = req.body(b.clone());
                }

                let t0 = Instant::now();
                match req.send().await {
                    Ok(resp) => {
                        let status = resp.status().as_u16();
                        // Drain body so keep-alive works.
                        let _ = resp.bytes().await;
                        let micros = t0.elapsed().as_micros() as u64;
                        acc.lock().await.record_success(micros, status);
                    }
                    Err(e) => {
                        acc.lock().await.record_error(e.to_string());
                    }
                }
            }
        }));
    }

    for h in handles {
        let _ = h.await;
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
