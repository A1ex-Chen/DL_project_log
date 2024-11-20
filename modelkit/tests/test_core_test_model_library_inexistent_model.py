def test_model_library_inexistent_model():
    with pytest.raises(ConfigurationNotFoundException):
        ModelLibrary(required_models=['model_that_does_not_exist'])
    configuration = {'existent_model': ModelConfiguration(model_type=Model,
        model_dependencies={'inexistent_model'})}
    with pytest.raises(ConfigurationNotFoundException):
        ModelLibrary(required_models=['existent_model'], configuration=
            configuration)
    p = ModelLibrary(required_models=['model_that_does_not_exist'],
        settings={'lazy_loading': True})
    with pytest.raises(ConfigurationNotFoundException):
        p.get('model_that_does_not_exist')
    with pytest.raises(ConfigurationNotFoundException):
        p.get('other_model_that_does_not_exist')
