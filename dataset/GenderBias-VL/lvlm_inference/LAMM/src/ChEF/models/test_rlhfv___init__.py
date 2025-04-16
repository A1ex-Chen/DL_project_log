def __init__(self, model_path, device='cuda', **kwargs):
    model, image_processor, image_token_len, tokenizer = init_muffin(model_path
        =model_path, device=device)
    self.model = model
    self.image_processor = image_processor
    self.image_token_len = image_token_len
    self.tokenizer = tokenizer
    self.model.eval()
    self.dtype = self.model.dtype
    self.device = self.model.device
    self.keywords = ['###']
