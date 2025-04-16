def test_sort_versions():
    assert SimpleDateAssetsVersioningSystem.sort_versions([
        '2021-11-15T17-30-56Z', '2020-11-15T17-30-56Z', '2021-10-15T17-30-56Z']
        ) == ['2021-11-15T17-30-56Z', '2021-10-15T17-30-56Z',
        '2020-11-15T17-30-56Z']
