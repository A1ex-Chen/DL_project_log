def test_increment_version():
    v1 = SimpleDateAssetsVersioningSystem.get_initial_version()
    time.sleep(2)
    v2 = SimpleDateAssetsVersioningSystem.increment_version()
    time.sleep(2)
    v3 = SimpleDateAssetsVersioningSystem.increment_version()
    assert SimpleDateAssetsVersioningSystem.sort_versions([v1, v2, v3]) == [v3,
        v2, v1]
