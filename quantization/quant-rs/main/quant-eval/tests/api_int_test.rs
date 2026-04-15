// @covers: saf/mod.rs
// Integration scope: using std::io and tempfile
use std::io as _;
use tempfile as _;
use quant-eval::*;

#[test]
fn e2e_run() { run(); }
#[test]
fn e2e_execute() { execute(); }
#[test]
fn e2e_init() { init(); }
#[test]
fn e2e_start() { start(); }
#[test]
fn e2e_stop() { stop(); }
#[test]
fn e2e_shutdown() { shutdown(); }
#[test]
fn e2e_quantize() { quantize(); }
#[test]
fn e2e_dequantize() { dequantize(); }
#[test]
fn e2e_service_init() { service_init(); }
#[test]
fn e2e_service_execute() { service_execute(); }
