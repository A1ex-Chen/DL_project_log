@skip_unless('ENABLE_TF_SERVING_TEST', 'True')
@skip_unless('ENABLE_TF_TEST', 'True')
def test_deploy_tf_models(monkeypatch):


    class DummyTFModel(TensorflowModel):
        CONFIGURATIONS = {'dummy_tf_model': {'asset': 'dummy_tf_model:0.0',
            'model_settings': {'output_dtypes': {'lambda': np.float32},
            'output_tensor_mapping': {'lambda': 'nothing'}, 'output_shapes':
            {'lambda': (3, 2, 1)}}}}
    with pytest.raises(ValueError):
        lib = ModelLibrary(models=[DummyTFModel], settings={'lazy_loading':
            True})
        deploy_tf_models(lib, 'remote', 'remote')
    ref = testing.ReferenceText(os.path.join(TEST_DIR, 'testdata',
        'tf_configs'))
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setenv('MODELKIT_ASSETS_DIR', tmp_dir)
        monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', TEST_DIR)
        monkeypatch.setenv('MODELKIT_STORAGE_PREFIX', 'testdata')
        monkeypatch.setenv('MODELKIT_STORAGE_PROVIDER', 'local')
        shutil.copytree(os.path.join(TEST_DIR, 'testdata'), os.path.join(
            tmp_dir, 'testdata'))
        os.makedirs(os.path.join(tmp_dir, 'testdata', 'dummy_tf_model_sub',
            '0.0'))
        lib = ModelLibrary(models=[DummyTFModel], settings={'lazy_loading':
            True})
        deploy_tf_models(lib, 'local-docker', 'local-docker')
        with open(os.path.join(tmp_dir, 'local-docker.config')) as f:
            ref.assert_equal('local-docker.config', f.read())
        deploy_tf_models(lib, 'remote', 'remote')
        with open(os.path.join(tmp_dir, 'remote.config')) as f:
            config_data = f.read().replace(TEST_DIR, 'STORAGE_BUCKET')
            ref.assert_equal('remote.config', config_data)
        deploy_tf_models(lib, 'local-process', 'local-process')
