def test_referencetext(monkeypatch):
    monkeypatch.setenv('UPDATE_REF', '0')
    with tempfile.TemporaryDirectory(prefix='common-') as tempdir:
        with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
            for i in range(3):
                print(f'value {i}', file=f)
        r = ReferenceText(tempdir)
        r.assert_equal('test.txt', ['value 0', 'value 1', 'value 2'])
        with pytest.raises(AssertionError):
            r.assert_equal('test.json', ['value 0'])
        with pytest.raises(AssertionError):
            r.assert_equal('fake.json', ['stuff'])
        r.assert_equal('test.json', ['value 0', 'value 1'], update_ref=True)
        with open(os.path.join(tempdir, 'test.json')) as f:
            assert f.read() == 'value 0\nvalue 1\n'
        r.assert_equal('test.json', 'value 0\nvalue 1')
        r.assert_equal('test.json', 'value 0\nvalue 2', update_ref=True)
        with open(os.path.join(tempdir, 'test.json')) as f:
            assert f.read() == 'value 0\nvalue 2\n'
