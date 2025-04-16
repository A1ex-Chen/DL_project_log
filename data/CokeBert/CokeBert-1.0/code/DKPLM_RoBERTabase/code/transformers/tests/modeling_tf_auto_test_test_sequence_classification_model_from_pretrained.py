def test_sequence_classification_model_from_pretrained(self):
    logging.basicConfig(level=logging.INFO)
    for model_name in ['bert-base-uncased']:
        config = AutoConfig.from_pretrained(model_name, force_download=True)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name
            , force_download=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TFBertForSequenceClassification)
