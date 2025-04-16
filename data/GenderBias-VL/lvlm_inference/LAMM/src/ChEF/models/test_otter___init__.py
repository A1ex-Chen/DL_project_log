def __init__(self, model_path, device='cuda', **kwargs) ->None:
    self.model = OtterForConditionalGeneration.from_pretrained(model_path)
    self.tokenizer = self.model.text_tokenizer
    self.image_processor = CLIPImageProcessor()
    self.tokenizer.padding_side = 'left'
    self.move_to_device(device)
