def __init__(self, image_encoder, text_encoder, num_classes):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.visual_encoder = image_encoder
    self.text_encoder = text_encoder
    hidden_size = text_encoder.config.hidden_size
    self.cls_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.
        ReLU(), nn.Linear(hidden_size, num_classes))
