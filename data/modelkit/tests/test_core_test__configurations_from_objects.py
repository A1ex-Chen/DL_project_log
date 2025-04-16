def test__configurations_from_objects():


    class SomeModel(Model):
        CONFIGURATIONS = {'yolo': {}, 'les simpsons': {}}


    class SomeModel2(Asset):
        CONFIGURATIONS = {'yolo2': {}, 'les simpsons2': {}}


    class ModelNoConf(Asset):
        pass
    configurations = _configurations_from_objects(SomeModel)
    assert 'yolo' in configurations
    assert 'les simpsons' in configurations
    configurations = _configurations_from_objects(ModelNoConf)
    assert {} == configurations
    configurations = _configurations_from_objects([SomeModel, SomeModel2,
        ModelNoConf])
    assert 'yolo' in configurations
    assert 'yolo2' in configurations
    assert 'les simpsons' in configurations
    assert 'les simpsons2' in configurations


    class Something:
        pass
    with pytest.raises(ValueError):
        _configurations_from_objects(Something)
