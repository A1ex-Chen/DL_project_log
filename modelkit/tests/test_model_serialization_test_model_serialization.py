def test_model_serialization():
    m = SomeModel()
    re_m = pickle.loads(pickle.dumps(m))
    assert re_m({'x': 1}).x == 1
