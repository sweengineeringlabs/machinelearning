use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llmmodel"))
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llmmodel-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn setup_mock_cache(models: &[(&str, Option<&str>)]) -> std::path::PathBuf {
    let cache = tempdir();
    for (model_id, config_json) in models {
        let dir_name = model_id.replace('/', "--");
        let model_dir = cache.join(&dir_name);
        std::fs::create_dir_all(&model_dir).unwrap();
        if let Some(config) = config_json {
            std::fs::write(model_dir.join("config.json"), config).unwrap();
        }
    }
    cache
}

// ── help ────────────────────────────────────────────────────────────

#[test]
fn help_flag() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("llmmodel"));
    assert!(stdout.contains("download"));
    assert!(stdout.contains("list"));
    assert!(stdout.contains("info"));
}

#[test]
fn download_subcommand_help() {
    let out = bin().args(["download", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--gguf"));
    assert!(stdout.contains("MODEL_ID"));
}

#[test]
fn list_subcommand_help() {
    let out = bin().args(["list", "--help"]).output().unwrap();
    assert!(out.status.success());
}

#[test]
fn info_subcommand_help() {
    let out = bin().args(["info", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("MODEL_ID"));
}

// ── global flags ────────────────────────────────────────────────────

#[test]
fn accepts_cache_dir_flag() {
    let cache = tempdir();
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
}

#[test]
fn accepts_token_flag() {
    let cache = tempdir();
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["--token", "hf_test_token", "list"])
        .output()
        .unwrap();
    assert!(out.status.success());
}

// ── list subcommand ─────────────────────────────────────────────────

#[test]
fn list_empty_cache() {
    let cache = tempdir();
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.trim().is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("No cached models found"));
}

#[test]
fn list_nonexistent_cache_dir() {
    let out = bin()
        .args([
            "--cache-dir",
            "/tmp/nonexistent_llmmodel_cache_xyz_12345",
            "list",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Cache directory does not exist"));
}

#[test]
fn list_shows_cached_models() {
    let cache = setup_mock_cache(&[
        ("openai-community/gpt2", Some("{}")),
        ("meta-llama/Llama-2-7b", Some("{}")),
    ]);
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("openai-community/gpt2"));
    assert!(stdout.contains("meta-llama/Llama-2-7b"));
}

#[test]
fn list_reconstructs_model_id_from_directory() {
    let cache = setup_mock_cache(&[("org/model-name", Some("{}"))]);
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("org/model-name"));
}

#[test]
fn list_shows_gguf_cached_models() {
    let cache = setup_mock_cache(&[("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", None)]);
    let model_dir = cache.join("TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF");
    std::fs::write(model_dir.join("tinyllama.Q4_0.gguf"), b"GGUF_DUMMY").unwrap();

    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"));
}

#[test]
fn list_shows_mixed_safetensors_and_gguf() {
    let cache = setup_mock_cache(&[
        ("openai-community/gpt2", Some("{}")),
        ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", None),
    ]);
    let gguf_dir = cache.join("TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF");
    std::fs::write(gguf_dir.join("tinyllama.Q4_0.gguf"), b"GGUF_DUMMY").unwrap();

    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["list"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("openai-community/gpt2"));
    assert!(stdout.contains("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"));
}

// ── info subcommand ─────────────────────────────────────────────────

#[test]
fn info_shows_config_json() {
    let config = r#"{"model_type":"gpt2","hidden_size":768,"num_hidden_layers":12}"#;
    let cache = setup_mock_cache(&[("openai-community/gpt2", Some(config))]);
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["info", "openai-community/gpt2"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"model_type\": \"gpt2\""));
    assert!(stdout.contains("\"hidden_size\": 768"));
    assert!(stdout.contains("\"num_hidden_layers\": 12"));
}

#[test]
fn info_pretty_prints_json() {
    let config = r#"{"a":1,"b":"two","c":[1,2,3]}"#;
    let cache = setup_mock_cache(&[("test/model", Some(config))]);
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["info", "test/model"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("  "));
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed["a"], 1);
    assert_eq!(parsed["b"], "two");
}

#[test]
fn info_uncached_model_fails() {
    let cache = tempdir();
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["info", "nonexistent/model"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not cached") || stderr.contains("nonexistent/model"));
}

#[test]
fn info_model_without_config_fails() {
    let cache = setup_mock_cache(&[("test/no-config", None)]);
    let out = bin()
        .args(["--cache-dir"])
        .arg(&cache)
        .args(["info", "test/no-config"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

// ── error cases ─────────────────────────────────────────────────────

#[test]
fn no_subcommand_shows_help() {
    let out = bin().output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Usage"));
}
