def __init__(self, image_encoder, text_encoder, num_classes, momentum=0.995,
    alpha=0.4, max_txt_len=40, use_distill=True):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.use_distill = use_distill
    self.visual_encoder = image_encoder
    self.text_encoder = text_encoder
    hidden_size = text_encoder.config.hidden_size
    self.cls_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.
        ReLU(), nn.Linear(hidden_size, num_classes))
    if self.use_distill:
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)
        self.cls_head_m = deepcopy(self.cls_head)
        self.momentum = momentum
        self.alpha = alpha
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m], [
            self.text_encoder, self.text_encoder_m], [self.cls_head, self.
            cls_head_m]]
        self.copy_params()
    self.max_txt_len = max_txt_len
