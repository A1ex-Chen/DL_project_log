@pytest.mark.slow
def test_model_from_pretrained(self):
    logging.basicConfig(level=logging.INFO)
    for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        config = BertConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, PretrainedConfig)
        model = BertModel.from_pretrained(model_name)
        model, loading_info = BertModel.from_pretrained(model_name,
            output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, PreTrainedModel)
        for value in loading_info.values():
            self.assertEqual(len(value), 0)
        config = BertConfig.from_pretrained(model_name, output_attentions=
            True, output_hidden_states=True)
        model = BertModel.from_pretrained(model_name, output_attentions=
            True, output_hidden_states=True)
        self.assertEqual(model.config.output_attentions, True)
        self.assertEqual(model.config.output_hidden_states, True)
        self.assertEqual(model.config, config)
