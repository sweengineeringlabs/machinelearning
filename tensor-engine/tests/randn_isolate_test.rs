//! Isolate RNG cost from tensor overhead.

use std::time::Instant;

#[test]
fn test_raw_rng_cost_200x200() {
    use rand_distr::{Distribution, StandardNormal};
    let numel = 200 * 200;
    let mut rng = rand::thread_rng();

    // Warmup
    let mut buf = vec![0.0f32; numel];
    for v in buf.iter_mut() { *v = StandardNormal.sample(&mut rng); }

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        for v in buf.iter_mut() {
            *v = StandardNormal.sample(&mut rng);
        }
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / iterations;
    eprintln!("Raw RNG fill 200x200 (no tensor): {:?} ({:.1} us)", per_call, per_call.as_nanos() as f64 / 1000.0);

    // Tensor creation overhead = randn time - raw RNG time
    eprintln!("Compare with randn 200x200 (~305us) to see tensor overhead");
}
