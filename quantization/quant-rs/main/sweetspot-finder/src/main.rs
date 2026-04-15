use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// Workspace Crates
use quant_api::{Quantizer, QuantFormat};
use quant_engine::DefaultQuantService;
use quant_eval::EvalService;
use quant_io::ModelIO;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(short, long)]
    input: String,
    #[arg(short, long, default_value = "32,64,128,256")]
    sweep: String,
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct SweepResult {
    block_size: usize,
    avg_snr_db: f32,
    avg_cosine: f32,
    overhead_pct: f32,
}

#[derive(Debug, serde::Serialize)]
struct FinderReport {
    model: String,
    results: Vec<SweepResult>,
    recommended_block_size: usize,
    sensitivity_ranking: Vec<LayerSensitivity>,
}

#[derive(Debug, serde::Serialize)]
struct LayerSensitivity { name: String, snr_db: f32 }

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let block_sizes: Vec<usize> = cli.sweep.split(',').map(|s| s.parse().unwrap()).collect();

    println!("🎯 Starting SweetSpot Sweep (Refactored Pipeline)");
    let tensors = ModelIO::load_safetensors(&cli.input)?;
    
    let mut sweep_results = Vec::new();
    let mut last_metrics = Vec::new();
    let multi_pb = MultiProgress::new();

    for &bs in &block_sizes {
        let pb = multi_pb.add(ProgressBar::new(tensors.len() as u64));
        pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [BS {msg}] [{bar:40.cyan/blue}] {pos}/{len}").unwrap());
        pb.set_message(bs.to_string());

        // Phase-1 MVP: only Int8 is implemented end-to-end. Other formats
        // will be added once their quantizer impls land with passing
        // round-trip tests; sweeping a block size is meaningful for any
        // block-based encoding so the sweep loop itself is unchanged.
        let service = DefaultQuantService::new(QuantFormat::Int8, bs);
        let eval = EvalService::new();

        let mut total_snr = 0.0_f32;
        let mut total_cosine = 0.0_f32;
        let mut snr_count = 0_usize;
        let mut current_metrics = Vec::new();

        for (name, weight) in &tensors {
            let quantized = service.quantize(weight)?;
            let dequantized = service.dequantize(&quantized)?;
            let m = eval.calculate_metrics(weight, &dequantized)?;
            // Skip +inf SNR (identical tensors) so it doesn't poison the average.
            if m.snr_db.is_finite() {
                total_snr += m.snr_db;
                snr_count += 1;
            }
            total_cosine += m.cosine;
            current_metrics.push((name.clone(), m.snr_db));
            pb.inc(1);
        }

        let n = tensors.len() as f32;
        let avg_snr_db = if snr_count > 0 {
            total_snr / snr_count as f32
        } else {
            f32::NAN
        };

        sweep_results.push(SweepResult {
            block_size: bs,
            avg_snr_db,
            avg_cosine: total_cosine / n,
            overhead_pct: (4.0 / bs as f32) * 100.0,
        });
        last_metrics = current_metrics;
        pb.finish_and_clear();
    }

    last_metrics.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let sensitivity_ranking: Vec<_> = last_metrics.iter().take(10).map(|(n, s)| LayerSensitivity { name: n.clone(), snr_db: *s }).collect();

    let report = FinderReport {
        model: cli.input,
        results: sweep_results,
        recommended_block_size: 128,
        sensitivity_ranking,
    };

    println!("\n{:<12} | {:<12} | {:<12}", "Block Size", "Avg SNR (dB)", "Overhead");
    println!("{:-<42}", "");
    for r in &report.results {
        println!("{:<12} | {:<12.2} | {:<12.2}%", r.block_size, r.avg_snr_db, r.overhead_pct);
    }

    if let Some(path) = cli.output {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, json)?;
    }

    Ok(())
}
