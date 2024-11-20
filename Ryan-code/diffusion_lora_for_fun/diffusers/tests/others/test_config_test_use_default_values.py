def test_use_default_values(self):
    config = SampleObject()
    config_dict = {k: v for k, v in config.config.items() if not k.
        startswith('_')}
    assert set(config_dict.keys()) == set(config.config._use_default_values)
    with tempfile.TemporaryDirectory() as tmpdirname:
        config.save_config(tmpdirname)
        config = SampleObject2.from_config(SampleObject2.load_config(
            tmpdirname))
        assert 'f' in config.config._use_default_values
        assert config.config.f == [1, 3]
    new_config = SampleObject4.from_config(config.config)
    assert new_config.config.f == [5, 4]
    config.config._use_default_values.pop()
    new_config_2 = SampleObject4.from_config(config.config)
    assert new_config_2.config.f == [1, 3]
    assert new_config_2.config.e == [1, 3]
