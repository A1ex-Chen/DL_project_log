@pytest.mark.parametrize('version, bump_major, major', [('1.0', True, '1'),
    ('2.1', False, '2')])
def test_get_update_cli_params(version, bump_major, major):
    res = MajorMinorAssetsVersioningSystem.get_update_cli_params(version=
        version, version_list=['1.1', '1.0', '0.1', '0.0'], bump_major=
        bump_major)
    assert res['params'] == {'bump_major': bump_major, 'major': major}
