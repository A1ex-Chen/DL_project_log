@pytest.mark.parametrize('item, model', [({'string': 'ok'},
    'some_complex_model'), ({'string': 'ok'}, 'async_model')])
def test_api_complex_type(item, model, api_no_type):
    res = api_no_type.post(f'/predict/{model}', headers={'Content-Type':
        'application/json'}, json=item)
    assert res.status_code == 200
    assert res.json()['sorted'] == ''.join(sorted(item['string']))
    res = api_no_type.post(f'/predict/batch/{model}', headers={
        'Content-Type': 'application/json'}, json=[item])
    assert res.status_code == 200
    assert res.json()[0]['sorted'] == ''.join(sorted(item['string']))
