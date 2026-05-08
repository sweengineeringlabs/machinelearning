use std::time::Instant;

use swe_llmmodel_model::LlmModel;

/// Run a decode warmup pass. Not fallible from the caller's POV —
/// failures are logged but don't block the load.
pub(crate) fn warmup(model: &mut LlmModel) {
    let start = Instant::now();
    if let Err(e) = model.warmup_decode() {
        log::warn!("  Decode warmup failed: {}", e);
    }
    log::info!("  Warmup: {:.0}ms", start.elapsed().as_secs_f64() * 1000.0);
}
