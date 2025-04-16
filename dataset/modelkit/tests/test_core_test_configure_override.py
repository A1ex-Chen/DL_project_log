def test_configure_override():


    class SomeModel(Model):
        CONFIGURATIONS = {'yolo': {'asset': 'ok/boomer'}, 'les simpsons': {}}


    class SomeOtherModel(Model):
        pass
    configurations = configure(models=SomeModel)
    assert configurations['yolo'].model_type == SomeModel
    assert configurations['yolo'].asset == 'ok/boomer'
    assert configurations['les simpsons'].model_type == SomeModel
    assert configurations['les simpsons'].asset is None
    configurations = configure(models=SomeModel, configuration={
        'somethingelse': ModelConfiguration(model_type=SomeOtherModel)})
    assert configurations['yolo'].model_type == SomeModel
    assert configurations['yolo'].asset == 'ok/boomer'
    assert configurations['les simpsons'].model_type == SomeModel
    assert configurations['les simpsons'].asset is None
    assert configurations['somethingelse'].model_type == SomeOtherModel
    assert configurations['somethingelse'].asset is None
    configurations = configure(models=SomeModel, configuration={'yolo':
        ModelConfiguration(model_type=SomeOtherModel)})
    assert configurations['yolo'].model_type == SomeOtherModel
    assert configurations['yolo'].asset is None
    configurations = configure(models=SomeModel, configuration={'yolo': {
        'asset': 'something/else'}})
    assert configurations['yolo'].model_type == SomeModel
    assert configurations['yolo'].asset == 'something/else'
    configurations = configure(models=SomeModel, configuration={'yolo2': {
        'model_type': SomeOtherModel}})
    assert configurations['yolo'].model_type == SomeModel
    assert configurations['yolo'].asset == 'ok/boomer'
    assert configurations['yolo2'].model_type == SomeOtherModel
    assert configurations['yolo2'].asset is None
