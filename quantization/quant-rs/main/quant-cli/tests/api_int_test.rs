// @covers: api
use quant_cli::*;
use quant_api::QuantFormat;
use quant_engine::DefaultQuantService;
use quant_eval::EvalService;
use quant_io::ModelIO;
use indicatif::ProgressBar;
use tempfile::tempdir;

#[test]
fn test_cli_integration_dependencies() {
    let _ = ModelIO::load_gguf_tensor("dummy.gguf", "test");
    let _format = QuantFormat::Nf4;
    let _service = DefaultQuantService::new(_format, 128);
    let _eval = EvalService::new();
    let _pb = ProgressBar::new(10);
    let _dir = tempdir().expect("integration scope");
}
