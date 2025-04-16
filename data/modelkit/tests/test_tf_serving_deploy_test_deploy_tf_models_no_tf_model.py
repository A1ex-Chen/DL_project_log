@skip_unless('ENABLE_TF_SERVING_TEST', 'True')
@skip_unless('ENABLE_TF_TEST', 'True')
def test_deploy_tf_models_no_tf_model():


    class DummyNonTFModel(Model):
        CONFIGURATIONS = {'dummy_non_tf_model': {}}
    lib = ModelLibrary(models=DummyNonTFModel, settings={'lazy_loading': True})
    deploy_tf_models(lib, 'local-docker')
