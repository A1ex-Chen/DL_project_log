def test_referencejson(monkeypatch):
    monkeypatch.setenv('UPDATE_REF', '0')
    with tempfile.TemporaryDirectory(prefix='common-') as tempdir:
        with open(os.path.join(tempdir, 'test.json'), 'w') as f:
            print('"value"', end='', file=f)
        r = ReferenceJson(tempdir)
        r.assert_equal('test.json', 'value')
        with pytest.raises(AssertionError):
            r.assert_equal('test.json', 'stuff')
        with pytest.raises(AssertionError):
            r.assert_equal('fake.json', 'stuff')
        r.assert_equal('test.json', 'new value', update_ref=True)
        with open(os.path.join(tempdir, 'test.json')) as f:
            assert f.read() == '"new value"'
        tempdir = os.path.join(tempdir, 'sub')
        r = ReferenceJson(tempdir)
        r.assert_equal('sub2/test.json', 'new value', update_ref=True)
        with open(os.path.join(tempdir, 'sub2/test.json')) as f:
            assert f.read() == '"new value"'
        r = ReferenceJson(tempdir)
        obj = {'date': datetime.date(2019, 1, 1), 'datetime': datetime.
            datetime(2019, 1, 1, tzinfo=datetime.timezone.utc), 'decimal':
            decimal.Decimal('0.1')}
        r.assert_equal('objects.json', obj, update_ref=True)
        with open(os.path.join(tempdir, 'objects.json')) as f:
            assert json.load(f) == {'date': '2019-01-01', 'datetime':
                '2019-01-01T00:00:00+00:00', 'decimal': '0.1'}
        with pytest.raises(TypeError, match='Unexpected function'):
            r.assert_equal('objects.json', lambda f: None)
