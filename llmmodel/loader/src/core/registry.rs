use swe_llmmodel_model::ModelBuilderRegistry;

/// Build the model registry with every supported architecture.
///
/// Registration order matters only for readability; lookup is
/// keyed by `config.architecture`. Aliases (e.g. `mistral` → llama,
/// `gemma2` → llama) share a builder because the architectures are
/// compatible.
pub fn create_registry() -> ModelBuilderRegistry {
    let mut reg = ModelBuilderRegistry::new();
    reg.register("llama", Box::new(swe_llmmodel_arch_llama::LlamaBuilder));
    reg.register("mistral", Box::new(swe_llmmodel_arch_llama::LlamaBuilder));
    reg.register("qwen2", Box::new(swe_llmmodel_arch_llama::LlamaBuilder));
    reg.register("phi3", Box::new(swe_llmmodel_arch_llama::LlamaBuilder));
    reg.register("gpt2", Box::new(swe_llmmodel_arch_gpt2::Gpt2Builder));
    reg.register("", Box::new(swe_llmmodel_arch_gpt2::Gpt2Builder)); // default
    reg.register("falcon", Box::new(swe_llmmodel_arch_falcon::FalconBuilder));
    reg.register("mixtral", Box::new(swe_llmmodel_arch_mixtral::MixtralBuilder));
    reg.register("gemma2", Box::new(swe_llmmodel_arch_llama::LlamaBuilder));
    reg.register("gemma3", Box::new(swe_llmmodel_arch_gemma3::Gemma3Builder));
    reg.register("gemma3_text", Box::new(swe_llmmodel_arch_gemma3::Gemma3Builder));
    reg.register("gemma4", Box::new(swe_llmmodel_arch_gemma4::Gemma4Builder));
    reg.register("gemma4_text", Box::new(swe_llmmodel_arch_gemma4::Gemma4Builder));
    reg.register("bert", Box::new(swe_llmmodel_arch_bert::BertBuilder));
    reg.register("nomic-bert", Box::new(swe_llmmodel_arch_nomic_bert::NomicBertBuilder));
    reg
}
