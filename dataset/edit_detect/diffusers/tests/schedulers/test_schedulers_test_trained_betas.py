def test_trained_betas(self):
    for scheduler_class in self.scheduler_classes:
        if scheduler_class in (VQDiffusionScheduler,
            CMStochasticIterativeScheduler):
            continue
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config, trained_betas=np.
            array([0.1, 0.3]))
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_pretrained(tmpdirname)
            new_scheduler = scheduler_class.from_pretrained(tmpdirname)
        assert scheduler.betas.tolist() == new_scheduler.betas.tolist()
