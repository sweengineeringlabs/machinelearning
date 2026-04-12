//! Regenerate `include/llmserv.h` from the Rust source on every build.
//!
//! Run `cargo build -p llmserv-ffi` and the header will be rewritten in
//! place. Commit the result so non-Rust consumers don't need a Rust
//! toolchain.

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = crate_dir.join("include").join("llmserv.h");

    // Only regenerate when source files change.
    println!("cargo:rerun-if-changed=main/src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let cfg = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("load cbindgen.toml");

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cfg)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&out);
        }
        Err(e) => {
            // Don't fail the build — header regeneration is a nice-to-have.
            // CI can enforce freshness with `git diff --exit-code`.
            println!("cargo:warning=cbindgen failed: {}", e);
        }
    }
}
