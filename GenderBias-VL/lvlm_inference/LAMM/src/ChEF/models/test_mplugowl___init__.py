def __init__(self, device, model_path, **kwargs):
    self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path,
        torch_dtype=torch.float32)
    self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
    self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
    self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
    self.device = device
    self.move_to_device(device)
    self.model.eval()
