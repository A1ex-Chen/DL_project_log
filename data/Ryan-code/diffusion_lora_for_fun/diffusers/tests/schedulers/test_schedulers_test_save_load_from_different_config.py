def test_save_load_from_different_config(self):
    obj = SchedulerObject()
    setattr(diffusers, 'SchedulerObject', SchedulerObject)
    logger = logging.get_logger('diffusers.configuration_utils')
    with tempfile.TemporaryDirectory() as tmpdirname:
        obj.save_config(tmpdirname)
        with CaptureLogger(logger) as cap_logger_1:
            config = SchedulerObject2.load_config(tmpdirname)
            new_obj_1 = SchedulerObject2.from_config(config)
        with open(os.path.join(tmpdirname, SchedulerObject.config_name), 'r'
            ) as f:
            data = json.load(f)
            data['unexpected'] = True
        with open(os.path.join(tmpdirname, SchedulerObject.config_name), 'w'
            ) as f:
            json.dump(data, f)
        with CaptureLogger(logger) as cap_logger_2:
            config = SchedulerObject.load_config(tmpdirname)
            new_obj_2 = SchedulerObject.from_config(config)
        with CaptureLogger(logger) as cap_logger_3:
            config = SchedulerObject2.load_config(tmpdirname)
            new_obj_3 = SchedulerObject2.from_config(config)
    assert new_obj_1.__class__ == SchedulerObject2
    assert new_obj_2.__class__ == SchedulerObject
    assert new_obj_3.__class__ == SchedulerObject2
    assert cap_logger_1.out == ''
    assert cap_logger_2.out == """The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and will be ignored. Please verify your config.json configuration file.
"""
    assert cap_logger_2.out.replace('SchedulerObject', 'SchedulerObject2'
        ) == cap_logger_3.out
