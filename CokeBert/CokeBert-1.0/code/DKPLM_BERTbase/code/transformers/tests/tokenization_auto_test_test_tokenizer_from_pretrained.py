@pytest.mark.slow
def test_tokenizer_from_pretrained(self):
    logging.basicConfig(level=logging.INFO)
    for model_name in list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())[:1]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.assertIsNotNone(tokenizer)
        self.assertIsInstance(tokenizer, BertTokenizer)
        self.assertGreater(len(tokenizer), 0)
    for model_name in list(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())[:1]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.assertIsNotNone(tokenizer)
        self.assertIsInstance(tokenizer, GPT2Tokenizer)
        self.assertGreater(len(tokenizer), 0)
