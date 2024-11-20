@skip_unless('ENABLE_TF_TEST', 'True')
def test_tf_model_local_path(dummy_tf_models):
    DummyTFModel, *_ = dummy_tf_models
    model = DummyTFModel(asset_path=os.path.join(TEST_DIR, 'testdata',
        'dummy_tf_model', '0.0'), model_settings={'output_dtypes': {
        'lambda': np.float32}, 'output_tensor_mapping': {'lambda':
        'nothing'}, 'output_shapes': {'lambda': (3, 2, 1)}})
    v = np.zeros((3, 2, 1), dtype=np.float32)
    assert np.allclose(v, model({'input_1': v})['lambda'])
