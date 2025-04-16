@pytest.fixture(scope='function')
def tf_serving(request, monkeypatch, working_dir, dummy_tf_models):
    DummyTFModel, *_ = dummy_tf_models
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', TEST_DIR)
    monkeypatch.setenv('MODELKIT_STORAGE_PREFIX', 'testdata')
    monkeypatch.setenv('MODELKIT_STORAGE_PROVIDER', 'local')
    lib = ModelLibrary(models=DummyTFModel, settings={'lazy_loading': True})
    yield tf_serving_fixture(request, lib, tf_version='2.8.0')
