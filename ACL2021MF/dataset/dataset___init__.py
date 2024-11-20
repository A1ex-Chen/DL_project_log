def __init__(self, config, json_path, tokenizer, copy_vocab,
    decoder_start_token_id, is_training=False, attachable_index=None):
    super(CommonGenDataset, self).__init__()
    self.config = config
    self.copy_vocab = copy_vocab
    self.tokenizer = tokenizer
    self.is_training = is_training
    self.decoder_start_token_id = decoder_start_token_id
    self.attachable_index = attachable_index
    np.set_printoptions(threshold=sys.maxsize)
    self.read_content(json_path)
