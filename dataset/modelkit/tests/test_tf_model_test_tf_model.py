@skip_unless('ENABLE_TF_SERVING_TEST', 'True')
@skip_unless('ENABLE_TF_TEST', 'True')
def test_tf_model(monkeypatch, working_dir, dummy_tf_models):
    DummyTFModel, *_ = dummy_tf_models
    monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', TEST_DIR)
    monkeypatch.setenv('MODELKIT_STORAGE_PREFIX', 'testdata')
    monkeypatch.setenv('MODELKIT_STORAGE_PROVIDER', 'local')
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    lib = ModelLibrary(models=DummyTFModel)
    assert not lib.settings.tf_serving.enable
    model = lib.get('dummy_tf_model')
    v = np.zeros((3, 2, 1), dtype=np.float32)
    assert np.allclose(v, model({'input_1': v})['lambda'])
