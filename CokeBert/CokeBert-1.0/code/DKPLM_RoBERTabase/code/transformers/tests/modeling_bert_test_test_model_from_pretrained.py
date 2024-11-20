@pytest.mark.slow
def test_model_from_pretrained(self):
    cache_dir = '/tmp/transformers_test/'
    for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        shutil.rmtree(cache_dir)
        self.assertIsNotNone(model)