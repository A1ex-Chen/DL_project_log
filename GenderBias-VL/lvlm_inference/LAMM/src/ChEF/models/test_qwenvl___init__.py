def __init__(self, model_path, device='cuda', **kwargs):
    self.model_config = QWenConfig.from_pretrained(model_path)
    self.model = QWenLMHeadModel.from_pretrained(model_path, device_map=device)
    self.tokenizer = QWenTokenizer.from_pretrained(model_path)
    self.model.eval()
    self.dtype = self.model.dtype
    self.device = self.model.device
    self.stop_words_ids = get_stop_words_ids(self.model.generation_config.
        chat_format, self.tokenizer)
    self.time = datetime.now().strftime('%H:%M:%S')
    self.image_path_temp = '/temp/' + self.time + '/'
    if not os.path.exists(self.image_path_temp):
        os.makedirs(self.image_path_temp)
