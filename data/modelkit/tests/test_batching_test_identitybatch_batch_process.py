@pytest.mark.parametrize('func,items,batch_size,expected', [(_identity, [],
    None, []), (_identity, [1], None, [1]), (_identity, list(range(2048)),
    None, list(range(2048))), (_double, [1, 2, 3], None, [2, 4, 6]), (
    _double, [1, 2, 3], 1, [2, 4, 6]), (_double, [1, 2, 3], 128, [2, 4, 6])])
def test_identitybatch_batch_process(func, items, batch_size, expected,
    monkeypatch):


    class SomeModel(Model):

        def _predict(self, item):
            return item
    m = SomeModel()
    monkeypatch.setattr(m, '_predict_batch', func)
    if batch_size:
        assert m.predict_batch(items, batch_size=batch_size) == expected
    else:
        assert m.predict_batch(items) == expected
