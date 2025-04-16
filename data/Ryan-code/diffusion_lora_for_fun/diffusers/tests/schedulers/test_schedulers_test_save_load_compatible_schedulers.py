def test_save_load_compatible_schedulers(self):
    SchedulerObject2._compatibles = ['SchedulerObject']
    SchedulerObject._compatibles = ['SchedulerObject2']
    obj = SchedulerObject()
    setattr(diffusers, 'SchedulerObject', SchedulerObject)
    setattr(diffusers, 'SchedulerObject2', SchedulerObject2)
    logger = logging.get_logger('diffusers.configuration_utils')
    with tempfile.TemporaryDirectory() as tmpdirname:
        obj.save_config(tmpdirname)
        with open(os.path.join(tmpdirname, SchedulerObject.config_name), 'r'
            ) as f:
            data = json.load(f)
            data['f'] = [0, 0]
            data['unexpected'] = True
        with open(os.path.join(tmpdirname, SchedulerObject.config_name), 'w'
            ) as f:
            json.dump(data, f)
        with CaptureLogger(logger) as cap_logger:
            config = SchedulerObject.load_config(tmpdirname)
            new_obj = SchedulerObject.from_config(config)
    assert new_obj.__class__ == SchedulerObject
    assert cap_logger.out == """The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and will be ignored. Please verify your config.json configuration file.
"""
