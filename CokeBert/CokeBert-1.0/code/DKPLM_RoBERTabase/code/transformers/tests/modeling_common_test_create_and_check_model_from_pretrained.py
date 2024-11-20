def create_and_check_model_from_pretrained(self):
    cache_dir = '/tmp/transformers_test/'
    for model_name in list(self.base_model_class.
        pretrained_model_archive_map.keys())[:1]:
        model = self.base_model_class.from_pretrained(model_name, cache_dir
            =cache_dir)
        shutil.rmtree(cache_dir)
        self.parent.assertIsNotNone(model)
