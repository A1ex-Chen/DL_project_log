def test_getattr_is_correct(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.dummy_attribute = 5
    model.register_to_config(test_attribute=5)
    logger = logging.get_logger('diffusers.models.modeling_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        assert hasattr(model, 'dummy_attribute')
        assert getattr(model, 'dummy_attribute') == 5
        assert model.dummy_attribute == 5
    assert cap_logger.out == ''
    logger = logging.get_logger('diffusers.models.modeling_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        assert hasattr(model, 'save_pretrained')
        fn = model.save_pretrained
        fn_1 = getattr(model, 'save_pretrained')
        assert fn == fn_1
    assert cap_logger.out == ''
    with self.assertWarns(FutureWarning):
        assert model.test_attribute == 5
    with self.assertWarns(FutureWarning):
        assert getattr(model, 'test_attribute') == 5
    with self.assertRaises(AttributeError) as error:
        model.does_not_exist
    assert str(error.exception
        ) == f"'{type(model).__name__}' object has no attribute 'does_not_exist'"
