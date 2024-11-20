@pytest.mark.slow
def test_model2model_from_pretrained(self):
    logging.basicConfig(level=logging.INFO)
    for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        model = Model2Model.from_pretrained(model_name)
        self.assertIsInstance(model.encoder, BertModel)
        self.assertIsInstance(model.decoder, BertForMaskedLM)
        self.assertEqual(model.decoder.config.is_decoder, True)
        self.assertEqual(model.encoder.config.is_decoder, False)
