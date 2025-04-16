def test_save_load_from_different_config_comp_schedulers(self):
    SchedulerObject3._compatibles = ['SchedulerObject', 'SchedulerObject2']
    SchedulerObject2._compatibles = ['SchedulerObject', 'SchedulerObject3']
    SchedulerObject._compatibles = ['SchedulerObject2', 'SchedulerObject3']
    obj = SchedulerObject()
    setattr(diffusers, 'SchedulerObject', SchedulerObject)
    setattr(diffusers, 'SchedulerObject2', SchedulerObject2)
    setattr(diffusers, 'SchedulerObject3', SchedulerObject3)
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(diffusers.logging.INFO)
    with tempfile.TemporaryDirectory() as tmpdirname:
        obj.save_config(tmpdirname)
        with CaptureLogger(logger) as cap_logger_1:
            config = SchedulerObject.load_config(tmpdirname)
            new_obj_1 = SchedulerObject.from_config(config)
        with CaptureLogger(logger) as cap_logger_2:
            config = SchedulerObject2.load_config(tmpdirname)
            new_obj_2 = SchedulerObject2.from_config(config)
        with CaptureLogger(logger) as cap_logger_3:
            config = SchedulerObject3.load_config(tmpdirname)
            new_obj_3 = SchedulerObject3.from_config(config)
    assert new_obj_1.__class__ == SchedulerObject
    assert new_obj_2.__class__ == SchedulerObject2
    assert new_obj_3.__class__ == SchedulerObject3
    assert cap_logger_1.out == ''
    assert cap_logger_2.out == """{'f'} was not found in config. Values will be initialized to default values.
"""
    assert cap_logger_3.out == """{'f'} was not found in config. Values will be initialized to default values.
"""
