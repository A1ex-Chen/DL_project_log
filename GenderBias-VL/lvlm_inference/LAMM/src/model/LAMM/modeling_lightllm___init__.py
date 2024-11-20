def __init__(self, batch_size, max_input_len, max_output_len, weight_dir,
    lora_path=None, lora_config: LoraConfig=None):
    super().__init__()
    self.model = LlamaModel(batch_size, max_input_len, max_output_len,
        weight_dir, lora_path=lora_path, lora_config=lora_config)
    self.infer_state: Dict = None
    with open(os.path.join(weight_dir, 'config.json'), 'r') as f:
        config_json = json.load(f)
    self.config = LlamaConfig(**config_json)
    with open(os.path.join(weight_dir, 'generation_config.json'), 'r') as f:
        generation_config_json = json.load(f)
    self.generation_config = GenerationConfig(**generation_config_json)
    self.device = torch.device('cuda')
    self.dtype = torch.float16
