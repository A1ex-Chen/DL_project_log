def test_model_serialization_without_types():
    m = SomeModelWithoutTypes()
    re_m = pickle.loads(pickle.dumps(m))
    assert re_m({'x': 1}) == {'x': 1}
