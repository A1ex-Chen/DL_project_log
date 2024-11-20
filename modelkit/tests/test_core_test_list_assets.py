def test_list_assets():


    class SomeModel(Asset):
        CONFIGURATIONS = {'model0': {'asset': 'some/asset'}}


    class SomeOtherModel(Asset):
        CONFIGURATIONS = {'model1': {'asset': 'some/asset'}}
    assert {'some/asset'} == list_assets(SomeModel)
    assert {'some/asset'} == list_assets([SomeModel, SomeOtherModel])
    assert {'some/asset', 'some/otherasset'} == list_assets([SomeModel,
        SomeOtherModel], configuration={'model1': {'asset': 'some/otherasset'}}
        )
    assert {'some/asset'} == list_assets([SomeModel, SomeOtherModel],
        required_models=['model0'], configuration={'model1': {'asset':
        'some/otherasset'}})
