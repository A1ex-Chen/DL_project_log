def test_register_to_config(self):
    obj = SampleObject()
    config = obj.config
    assert config['a'] == 2
    assert config['b'] == 5
    assert config['c'] == (2, 5)
    assert config['d'] == 'for diffusion'
    assert config['e'] == [1, 3]
    obj = SampleObject(_name_or_path='lalala')
    config = obj.config
    assert config['a'] == 2
    assert config['b'] == 5
    assert config['c'] == (2, 5)
    assert config['d'] == 'for diffusion'
    assert config['e'] == [1, 3]
    obj = SampleObject(c=6)
    config = obj.config
    assert config['a'] == 2
    assert config['b'] == 5
    assert config['c'] == 6
    assert config['d'] == 'for diffusion'
    assert config['e'] == [1, 3]
    obj = SampleObject(1, c=6)
    config = obj.config
    assert config['a'] == 1
    assert config['b'] == 5
    assert config['c'] == 6
    assert config['d'] == 'for diffusion'
    assert config['e'] == [1, 3]
