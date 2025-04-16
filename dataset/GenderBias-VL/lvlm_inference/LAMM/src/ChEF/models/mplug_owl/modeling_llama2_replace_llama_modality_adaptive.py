def replace_llama_modality_adaptive():
    transformers.models.llama.configuration_llama.LlamaConfig = LlamaConfig
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
        LlamaDecoderLayer)
    transformers.models.llama.modeling_llama.LlamaModel.forward = model_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = (
        causal_model_forward)
