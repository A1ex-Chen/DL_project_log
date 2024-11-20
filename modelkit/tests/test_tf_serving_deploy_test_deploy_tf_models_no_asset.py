@skip_unless('ENABLE_TF_SERVING_TEST', 'True')
@skip_unless('ENABLE_TF_TEST', 'True')
def test_deploy_tf_models_no_asset():
    np = pytest.importorskip('numpy')


    class DummyTFModelNoAsset(TensorflowModel):
        CONFIGURATIONS = {'dummy_non_tf_model': {'model_settings': {
            'output_dtypes': {'lambda': np.float32},
            'output_tensor_mapping': {'lambda': 'nothing'}, 'output_shapes':
            {'lambda': (3, 2, 1)}}}}
    lib = ModelLibrary(models=DummyTFModelNoAsset, settings={'lazy_loading':
        True})
    with pytest.raises(ValueError):
        deploy_tf_models(lib, 'local-docker')
