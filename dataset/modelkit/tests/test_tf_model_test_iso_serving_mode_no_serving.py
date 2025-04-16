@skip_unless('ENABLE_TF_TEST', 'True')
def test_iso_serving_mode_no_serving(dummy_tf_models, monkeypatch, working_dir
    ):
    monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', TEST_DIR)
    monkeypatch.setenv('MODELKIT_STORAGE_PREFIX', 'testdata')
    monkeypatch.setenv('MODELKIT_STORAGE_PROVIDER', 'local')
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    monkeypatch.setenv('MODELKIT_TF_SERVING_ATTEMPTS', 1)
    with pytest.raises(grpc.RpcError):
        ModelLibrary(required_models=['dummy_tf_model'], settings=
            LibrarySettings(tf_serving={'enable': True, 'port': 8500,
            'mode': 'grpc', 'host': 'localhost'}), models=dummy_tf_models)
    with pytest.raises(requests.exceptions.ConnectionError):
        ModelLibrary(required_models=['dummy_tf_model'], settings=
            LibrarySettings(tf_serving={'enable': True, 'port': 8501,
            'mode': 'rest', 'host': 'localhost'}), models=dummy_tf_models)
