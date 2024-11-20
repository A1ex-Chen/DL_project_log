def test_from_pretrained(self):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_pretrained(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
        scheduler_config = dict(scheduler.config)
        del scheduler_config['_use_default_values']
        assert scheduler_config == new_scheduler.config
