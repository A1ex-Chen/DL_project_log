def __init__(self, device, model_path, **kwargs):
    self.model_name = get_model_name_from_path(model_path)
    self.tokenizer, self.model, self.image_processor, self.context_len = (
        load_pretrained_model(model_path, None, self.model_name, load_8bit=
        False, load_4bit=False, device=device))
    self.device = device
    self.move_to_device(device)
    self.model.eval()
    self.tokenizer.padding_side = 'left'
    if not hasattr(self.tokenizer, 'pad_token_id'):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_i
