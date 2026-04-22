# GEMINI.md

This document provides an overview of the `machinelearning` project, designed to serve as instructional context for future interactions with the Gemini CLI.

## Project Overview

The `machinelearning` project is a pure-Rust ML platform comprising three main Cargo workspaces and a developer-tools tree. Its primary goal is to provide foundational ML capabilities, model acquisition and I/O, LLM serving, and tools for tasks like model quantization.

The project is structured with a clear dependency flow:
*   `llmserv` depends on `llmmodel`.
*   `llmmodel` depends on `main` (foundations).
*   `main` (foundations) is the root workspace.

The `devtools/quantize` component is a standalone tool that reaches into `llmserv` for its quantizer library, representing a single backward dependency.

## Building and Running

The project uses Cargo workspaces for organization. Below are the commands to build and run various components.

### Build Everything

To build the different workspaces:

```bash
# Root workspace (foundations + devtools)
cargo build --release

# llmmodel workspace (download / io / weights / cli)
cargo build --release --manifest-path llmmodel/Cargo.toml

# llmserv workspace (inference stack, daemon, cli, embedding server)
cargo build --release --manifest-path llmserv/Cargo.toml
```

### Running the LLM Daemon (`swellmd`)

The `swellmd` daemon is configuration-driven. Edit `llmserv/main/config/application.toml` or set an override at `$XDG_CONFIG_HOME/llmserv/application.toml`.

Example configuration snippet:

```toml
[server]
host = "127.0.0.1"
port = 8080

[model]
source = "safetensors"                  # or "gguf"
id = "google/gemma-3-1b-it"             # HuggingFace repo ID
# path = "./models/gemma-3-1b-it-Q4_0.gguf"   # when source = "gguf"
```

To run the daemon:

```bash
llmserv/target/release/swellmd
# → serves OpenAI-compatible /v1/chat/completions on configured host/port
```

For gated HuggingFace models, set `HF_TOKEN=hf_...` in the environment.

### Quantizing a Model (`rustml-quantize`)

This is an optional, offline tool for converting SafeTensors models to GGUF.

```bash
cargo run --release -p rustml-quantize -- 
  --model google/gemma-3-1b-it 
  --target q4_0 
  --output ./models/gemma-3-1b-it-q4_0.gguf
```

Refer to `devtools/quantize/README.md` for more details.

### Running the Embedding Server (`swe-ml-embed`)

Configure the embedding server via `llmserv/main/config/application.toml` or `$XDG_CONFIG_HOME/llmserv/application.toml`.

Example configuration snippet:

```toml
[embedding.server]
host = "127.00.1"
port = 8081

[embedding.model]
gguf_path = "./models/nomic-embed-text-v1.5.Q8_0.gguf"
```

To run the embedding server:

```bash
llmserv/target/release/swe-ml-embed
# → serves OpenAI-compatible /v1/embeddings on configured host/port
```

### Developer CLI (`sweai`)

The `sweai` CLI is flag-driven and provides various subcommands:

```bash
llmserv/target/release/sweai --help
```

Available subcommands include `infer`, `hub`, `gguf`, `tokenizer`.

## Development Conventions

This project is built with Rust. Adherence to Rust's idiomatic practices and coding standards is expected. Specific coding styles, testing practices, or contribution guidelines are detailed within the project's documentation.

## Key Documentation

The project includes extensive documentation covering various aspects:

| Topic                     | Location                                                    |
| :------------------------ | :---------------------------------------------------------- |
| Daemon architecture       | `llmserv/main/features/daemon/docs/3-design/architecture.md` |
| Load testing strategy     | `docs/5-testing/load_testing_strategy.md`                   |
| Load testing reports      | `docs/5-testing/report/load_testing_*.md`                   |
| Performance reports       | `docs/5-testing/report/perf-*.md`                           |
| Operations guide          | `docs/7-operations/operations_guide.md`                     |
| Deployment guide          | `docs/6-deployment/deployment_guide.md`                     |
| Developer guide           | `docs/4-development/developer_guide.md`                     |
| Inference pipeline        | `docs/3-design/guides/inference_pipeline.md`                |
| Glossary                  | `docs/3-design/glossary.md`                                 |

## License

The project is licensed under MIT OR Apache-2.0.

---

## Analysis of `llmmodel` and `llmserv`

### **`llmmodel`**

*   **High-Level Responsibility:** The `llmmodel` workspace is primarily responsible for model acquisition, handling, and fundamental machine learning operations. It serves as the core library for interacting with various Large Language Models (LLMs), managing their lifecycle from download to basic processing.

*   **Internal Organization:**
    *   **Model Acquisition & I/O:** Contains crates for downloading models (e.g., from HuggingFace via `hf-hub`), managing I/O operations (`io`), and handling model weights (`weights`), including specific mappers for different architectures.
    *   **Core ML Functionality:** Includes modules for essential ML operations such as `kernel` (low-level numerical computations), `layers` (neural network layers), `gguf` (handling the GGUF model format), `tokenizer` (text tokenization), and `quantizer` (model quantization).
    *   **Architecture Support:** Provides dedicated sub-crates under `arch/` for various LLM architectures, including `llama`, `gpt2`, `falcon`, `mixtral`, `gemma3`, `gemma4`, `bert`, and `nomic_bert`, demonstrating broad compatibility.
    *   **Command-Line Interface (CLI):** The `cli` crate offers a command-line tool for basic operations like downloading, listing, and querying information about models.
    *   **Dependencies:** Relies on foundational crates from the root `main/features` workspace (`swe-ml-tensor`, `swe-ml-normalization`, `swe-cli`) for basic ML primitives. It also utilizes several external Rust crates for asynchronous operations (`tokio`), serialization (`serde`), HTTP requests (`reqwest`), CLI parsing (`clap`), memory mapping (`memmap2`), and numerical processing (`rayon`, `half`).

### **`llmserv`**

*   **High-Level Responsibility:** The `llmserv` workspace is dedicated to serving LLMs, providing the inference-server stack. It encapsulates the compute library necessary for inference and various frontends for exposing LLM capabilities via APIs.

*   **Internal Organization:**
    *   **Inference Compute Stack:** This is the heart of `llmserv`, located under `main/features/inference/`. It includes crates for `generation` (text generation), `prefill` (initial token processing), `compute` (inference computations), `thread-config` (thread management for inference), `backend-api`, and `backend-llama-cpp` (potential integration with `llama.cpp` for specialized inference).
    *   **Serving Frontends:**
        *   `main/features/inference/systemd`: Likely the LLM daemon (`swellmd`) responsible for handling chat completions.
        *   `main/features/embedding/systemd`: The embedding HTTP server (`swe-ml-embed`).
        *   `main/features/inference/cli`: A multi-command developer CLI (`llmc`) for interacting with the serving capabilities.
    *   **Shared Infrastructure:** `main/features/systemd` likely provides common services for daemon applications, such as configuration and logging.
    *   **Foreign Function Interface (FFI):** The `main/features/inference/ffi` crate suggests capabilities for integrating with other languages or environments, potentially for desktop or IDE applications.
    *   **Experimentation:** The `main/features/experimentation/llmforge` is explicitly excluded, indicating it's an archived or deprecated prototype.
    *   **Dependencies:** `llmserv` depends on the same foundational crates from `main/features` as `llmmodel` (e.g., `swe-ml-tensor`, `swe-ml-normalization`). Crucially, it has extensive dependencies on many crates from the `llmmodel` workspace (e.g., `swe-llmmodel-download`, `swe-llmmodel-io`, `swe-llmmodel-weights`, `swe-llmmodel-tokenizer`, and all `swe-llmmodel-arch-*` crates), confirming the stated architectural dependency. Additionally, it uses `axum` for building web services, `toml` for configuration, and `llama-cpp-2` for `llama.cpp` bindings.

### **Interaction and Architectural Patterns**

The interaction between `llmmodel` and `llmserv` strictly follows the layered architectural pattern described in the `README.md`: **`llmserv` → `llmmodel` → `main` (foundations)**.

*   **`llmmodel` as a Library Layer:** `llmmodel` functions as a lower-level library, providing all the necessary components for understanding, loading, and performing basic operations on LLM architectures. It abstracts away the complexities of model file formats (like GGUF), weight handling, tokenization, and quantization.

*   **`llmserv` as an Application/Service Layer:** `llmserv` builds upon the capabilities exposed by `llmmodel`. It consumes the processed models and data from `llmmodel` and integrates them into a high-performance serving infrastructure. `llmserv` focuses on the orchestration of inference, exposing user-friendly APIs (e.g., OpenAI-compatible chat completions and embedding endpoints) via web servers built with `axum`.

This layered approach promotes modularity and separation of concerns. `llmmodel` focuses on model data and computation primitives, while `llmserv` focuses on deploying and exposing these models as services.