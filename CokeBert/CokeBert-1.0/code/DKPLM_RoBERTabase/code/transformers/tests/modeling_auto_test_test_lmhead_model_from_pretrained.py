@pytest.mark.slow
def test_lmhead_model_from_pretrained(self):
    logging.basicConfig(level=logging.INFO)
    for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)
        model = AutoModelWithLMHead.from_pretrained(model_name)
        model, loading_info = AutoModelWithLMHead.from_pretrained(model_name,
            output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForMaskedLM)
