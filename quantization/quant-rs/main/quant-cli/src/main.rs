use clap::{Parser, ValueEnum};
use std::collections::{HashMap, HashSet};
use indicatif::{ProgressBar, ProgressStyle};
use candle_core::{Tensor, Device, DType};

// Workspace Crates
use quant_api::{Quantizer, QuantFormat};
use quant_engine::DefaultQuantService;
use quant_eval::EvalService;
use quant_io::ModelIO;

#[derive(Parser, Debug)]
#[command(author, version, about = "WeightScope CLI", long_about = None)]
struct Cli {
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    output: Option<String>,
    #[arg(short, long, value_enum, default_value_t = QuantFormatCli::Nf4)]
    format: QuantFormatCli,
    #[arg(short, long, default_value_t = 128)]
    block_size: usize,
    #[arg(long, default_value_t = true)]
    eval: bool,
    #[arg(long)]
    reference: Option<String>,
    #[arg(long)]
    protect_report: Option<String>,
    #[arg(long, default_value_t = 5)]
    protect_count: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum QuantFormatCli { Int8, Nf4, Int4, Fp16, Q4_0 }

impl From<QuantFormatCli> for QuantFormat {
    fn from(val: QuantFormatCli) -> Self {
        match val {
            QuantFormatCli::Int8 => QuantFormat::Int8,
            QuantFormatCli::Nf4 => QuantFormat::Nf4,
            QuantFormatCli::Int4 => QuantFormat::Int4,
            QuantFormatCli::Fp16 => QuantFormat::Fp16,
            QuantFormatCli::Q4_0 => QuantFormat::Q4_0,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    println!("🚀 Starting WeightScope CLI (Refactored Pipeline)");

    if cli.input.ends_with(".gguf") {
        return run_gguf_analysis(&cli);
    }

    run_quantization(&cli)
}

fn run_gguf_analysis(cli: &Cli) -> anyhow::Result<()> {
    println!("📂 Mode: GGUF Analysis");
    let ref_path = cli.reference.as_ref().expect("GGUF analysis requires a --reference Safetensors file");
    let ref_tensors = ModelIO::load_safetensors(ref_path)?;
    let eval_service = EvalService::new();
    
    for (name, ref_weight) in ref_tensors.iter().take(10) {
        // GGUF names usually need mapping, but for this test we'll try direct
        if let Ok(dequantized) = ModelIO::load_gguf_tensor(&cli.input, name) {
            let metrics = eval_service.calculate_metrics(ref_weight, &dequantized)?;
            println!("   - {}: SNR={:.2}dB", name, metrics.snr_db);
        }
    }
    Ok(())
}

fn run_quantization(cli: &Cli) -> anyhow::Result<()> {
    let out_path = cli.output.as_ref().expect("Output path required");
    let tensors = ModelIO::load_safetensors(&cli.input)?;
    let service = DefaultQuantService::new(cli.format.into(), cli.block_size);
    let eval_service = EvalService::new();

    let mut protected_layers = HashSet::new();
    if let Some(path) = &cli.protect_report {
        let file = std::fs::File::open(path)?;
        let json: serde_json::Value = serde_json::from_reader(file)?;
        if let Some(ranking) = json.get("sensitivity_ranking").and_then(|r| r.as_array()) {
            for entry in ranking.iter().take(cli.protect_count) {
                if let Some(n) = entry.get("name").and_then(|n| n.as_str()) {
                    protected_layers.insert(n.to_string());
                }
            }
        }
    }

    let mut new_tensors = HashMap::new();
    let pb = ProgressBar::new(tensors.len() as u64);

    for (name, weight) in tensors {
        if protected_layers.contains(&name) {
            let q_data = weight.flatten_all()?.to_vec1::<f32>()?;
            let bytes = bytemuck::cast_slice(&q_data).to_vec();
            let dtype = match weight.dtype() {
                DType::BF16 => safetensors::Dtype::BF16,
                DType::F16 => safetensors::Dtype::F16,
                _ => safetensors::Dtype::F32,
            };
            new_tensors.insert(name, (dtype, weight.shape().dims().to_vec(), bytes));
            pb.inc(1);
            continue;
        }

        let quantized = service.quantize(&weight)?;
        
        if cli.eval {
            let dequantized = service.dequantize(&quantized)?;
            let _metrics = eval_service.calculate_metrics(&weight, &dequantized)?;
        }

        let q_bytes = quantized.data.flatten_all()?.to_vec1::<u8>()?;
        new_tensors.insert(name.clone(), (safetensors::Dtype::U8, vec![q_bytes.len()], q_bytes));

        let scale_data = quantized.scale.flatten_all()?.to_vec1::<f32>()?;
        new_tensors.insert(format!("{}.scales", name), (safetensors::Dtype::F32, quantized.scale.shape().dims().to_vec(), bytemuck::cast_slice(&scale_data).to_vec()));
        
        pb.inc(1);
    }

    pb.finish_with_message("Done!");
    ModelIO::save_safetensors(out_path, new_tensors)?;
    println!("✅ Model saved: {}", out_path);
    Ok(())
}
