def test_api_doc(api_no_type):
    r = testing.ReferenceJson(os.path.join(TEST_DIR, 'testdata', 'api'))
    res = api_no_type.get('/openapi.json')
    if platform.system() != 'Windows':
        r.assert_equal('openapi.json', _strip_description_fields(res.json()))
