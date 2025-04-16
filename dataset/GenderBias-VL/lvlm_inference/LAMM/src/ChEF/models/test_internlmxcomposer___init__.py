def __init__(self, model_path, device='cuda', **kwargs):
    self.model_config = InternLMXcomposer2Config.from_pretrained(model_path)
    self.model = InternLMXComposer2ForCausalLM.from_pretrained(model_path,
        trust_remote_code=True, device_map=device)
    self.tokenizer = InternLMXComposer2Tokenizer.from_pretrained(model_path,
        trust_remote_code=True)
    self.model.tokenizer = self.tokenizer
    self.model.eval()
    self.dtype = self.model.dtype
    self.device = self.model.device
