def test_get_update_cli_params():
    res = SimpleDateAssetsVersioningSystem.get_update_cli_params(version_list
        =['2021-11-15T17-30-56Z', '2021-10-15T17-30-56Z',
        '2020-11-15T17-30-56Z'])
    assert res['params'] == {}
