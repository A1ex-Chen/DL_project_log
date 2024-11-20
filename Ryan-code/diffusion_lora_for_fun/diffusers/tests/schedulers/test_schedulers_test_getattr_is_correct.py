def test_getattr_is_correct(self):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        scheduler.dummy_attribute = 5
        scheduler.register_to_config(test_attribute=5)
        logger = logging.get_logger('diffusers.configuration_utils')
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(scheduler, 'dummy_attribute')
            assert getattr(scheduler, 'dummy_attribute') == 5
            assert scheduler.dummy_attribute == 5
        assert cap_logger.out == ''
        logger = logging.get_logger('diffusers.schedulers.scheduling_utils')
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(scheduler, 'save_pretrained')
            fn = scheduler.save_pretrained
            fn_1 = getattr(scheduler, 'save_pretrained')
            assert fn == fn_1
        assert cap_logger.out == ''
        with self.assertWarns(FutureWarning):
            assert scheduler.test_attribute == 5
        with self.assertWarns(FutureWarning):
            assert getattr(scheduler, 'test_attribute') == 5
        with self.assertRaises(AttributeError) as error:
            scheduler.does_not_exist
        assert str(error.exception
            ) == f"'{type(scheduler).__name__}' object has no attribute 'does_not_exist'"
