//! Build script — generates Rust types for `proto/embed.proto`.
//!
//! Output is written into `OUT_DIR` and re-included from
//! `src/api/proto.rs` via `include!`.  We pin re-build triggers to
//! the proto file itself so iterating on Rust source is fast.

fn main() {
    println!("cargo:rerun-if-changed=proto/embed.proto");
    println!("cargo:rerun-if-changed=build.rs");

    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .compile_protos(&["proto/embed.proto"], &["proto"])
        .expect("failed to compile embed.proto");
}
