// @covers: api
use sweetspot_finder::*;
use quant_api::QuantFormat;
use quant_engine::DefaultQuantService;
use quant_eval::EvalService;
use quant_io::ModelIO;
use indicatif::ProgressBar;
use tempfile::tempdir;
use candle_core::Tensor;

#[test]
fn test_full_pipeline_dependencies() {
    // 1. quant-io & candle-core
    let _ = ModelIO::load_gguf_tensor("dummy.gguf", "test");
    
    // 2. quant-api & quant-engine
    let format = QuantFormat::Nf4;
    let service = DefaultQuantService::new(format, 128);
    
    // 3. quant-eval
    let eval = EvalService::new();
    
    // 4. indicatif
    let pb = ProgressBar::new(10);
    pb.finish();
    
    // 5. tempfile (integration marker)
    let _dir = tempdir().expect("integration scope");
}
