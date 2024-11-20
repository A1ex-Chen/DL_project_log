def test_model_from_pretrained(self):
    import h5py
    self.assertTrue(h5py.version.hdf5_version.startswith('1.10'))
    logging.basicConfig(level=logging.INFO)
    for model_name in ['bert-base-uncased']:
        config = AutoConfig.from_pretrained(model_name, force_download=True)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)
        model = TFAutoModel.from_pretrained(model_name, force_download=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TFBertModel)
