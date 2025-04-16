def test_save_load(self):
    obj = SampleObject()
    config = obj.config
    assert config['a'] == 2
    assert config['b'] == 5
    assert config['c'] == (2, 5)
    assert config['d'] == 'for diffusion'
    assert config['e'] == [1, 3]
    with tempfile.TemporaryDirectory() as tmpdirname:
        obj.save_config(tmpdirname)
        new_obj = SampleObject.from_config(SampleObject.load_config(tmpdirname)
            )
        new_config = new_obj.config
    config = dict(config)
    new_config = dict(new_config)
    assert config.pop('c') == (2, 5)
    assert new_config.pop('c') == [2, 5]
    config.pop('_use_default_values')
    assert config == new_config
