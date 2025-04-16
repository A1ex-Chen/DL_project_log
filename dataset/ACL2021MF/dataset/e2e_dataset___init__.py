def __init__(self, config, json_path, tokenizer, copy_vocab, is_training=False
    ):
    super(E2EDataset, self).__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.is_training = is_training
    self.copy_vocab = copy_vocab
    np.set_printoptions(threshold=sys.maxsize)
    self.keyword_norm = {'eatType': 'type', 'familyFriendly':
        'family friendly', 'priceRange': 'price range'}
    self.key_index = {'type': 1, 'family friendly': 3, 'price range': 4,
        'near': 5, 'name': 6, 'food': 7, 'area': 8, 'customer rating': 9}
    self.read_content(json_path)
