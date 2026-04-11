use clap::Parser;
use std::path::PathBuf;

use rustml_quantize::{QuantizeConfig, QuantTarget, QuantizeEngine, create_engine};

/// Quantize a HuggingFace SafeTensors model to GGUF format.
#[derive(Parser, Debug)]
#[command(name = "rustml-quantize")]
#[command(about = "Convert SafeTensors models to quantized GGUF format")]
struct Args {
    /// HuggingFace model ID (e.g., google/gemma-4-e2b-it)
    #[arg(long)]
    model: String,

    /// Target quantization format: q8_0, q4_0, q4_1
    #[arg(long, default_value = "q8_0")]
    target: String,

    /// Output GGUF file path
    #[arg(long)]
    output: PathBuf,

    /// Keep the output/lm_head layer in F32
    #[arg(long)]
    preserve_output: bool,

    /// Minimum dimension to quantize (skip smaller tensors)
    #[arg(long, default_value_t = 0)]
    min_dim: usize,

    /// Show per-tensor quantization error metrics
    #[arg(long)]
    metrics: bool,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    let args = Args::parse();

    let target = QuantTarget::from_str(&args.target).unwrap_or_else(|| {
        eprintln!("Invalid target: {}. Supported: q8_0, q4_0, q4_1", args.target);
        std::process::exit(1);
    });

    let config = QuantizeConfig {
        model_id: args.model,
        target,
        output_path: args.output,
        min_dim: args.min_dim,
        preserve_output: args.preserve_output,
        show_metrics: args.metrics,
    };

    let engine = create_engine();
    match engine.run(&config) {
        Ok(report) => {
            println!("\nQuantization complete!");
            println!("  Tensors: {} quantized, {} skipped", report.quantized_tensors, report.skipped_tensors);
            println!("  Size: {} -> {} ({:.2}x compression)",
                format_bytes(report.original_bytes),
                format_bytes(report.quantized_bytes),
                report.compression_ratio,
            );
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{bytes} B")
    }
}
