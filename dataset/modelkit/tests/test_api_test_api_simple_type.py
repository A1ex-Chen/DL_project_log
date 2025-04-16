@pytest.mark.parametrize('item', ['ok', 'ko'])
def test_api_simple_type(item, api_no_type):
    res = api_no_type.post('/predict/some_model', headers={'Content-Type':
        'application/json'}, json=item)
    assert res.status_code == 200
    assert res.json() == item
    res = api_no_type.post('/predict/batch/some_model', headers={
        'Content-Type': 'application/json'}, json=[item])
    assert res.status_code == 200
    assert res.json() == [item]
