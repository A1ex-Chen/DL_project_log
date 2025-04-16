def test_compatibles(self):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        assert all(c is not None for c in scheduler.compatibles)
        for comp_scheduler_cls in scheduler.compatibles:
            comp_scheduler = comp_scheduler_cls.from_config(scheduler.config)
            assert comp_scheduler is not None
        new_scheduler = scheduler_class.from_config(comp_scheduler.config)
        new_scheduler_config = {k: v for k, v in new_scheduler.config.items
            () if k in scheduler.config}
        scheduler_diff = {k: v for k, v in new_scheduler.config.items() if 
            k not in scheduler.config}
        assert new_scheduler_config == dict(scheduler.config)
        init_keys = inspect.signature(scheduler_class.__init__
            ).parameters.keys()
        assert set(scheduler_diff.keys()).intersection(set(init_keys)) == set()
