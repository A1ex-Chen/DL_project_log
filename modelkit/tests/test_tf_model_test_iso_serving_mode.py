@skip_unless('ENABLE_TF_SERVING_TEST', 'True')
@skip_unless('ENABLE_TF_TEST', 'True')
@pytest.mark.parametrize('model_name, test_items', [('dummy_tf_model',
    TEST_ITEMS_IS_EMPTY), ('dummy_tf_model_no_is_empty', TEST_ITEMS)])
def test_iso_serving_mode(model_name, test_items, tf_serving, dummy_tf_models):
    lib_serving_grpc = ModelLibrary(required_models=[model_name], settings=
        LibrarySettings(tf_serving={'enable': True, 'port': 8500, 'mode':
        'grpc', 'host': 'localhost'}), models=dummy_tf_models)
    model_grpc = lib_serving_grpc.get(model_name)
    lib_serving_rest = ModelLibrary(required_models=[model_name], settings=
        LibrarySettings(tf_serving={'enable': True, 'port': 8501, 'mode':
        'rest', 'host': 'localhost'}), models=dummy_tf_models)
    model_rest = lib_serving_rest.get(model_name)
    lib_tflib = ModelLibrary(required_models=[model_name], settings=
        LibrarySettings(), models=dummy_tf_models)
    assert not lib_tflib.settings.tf_serving.enable
    model_tflib = lib_tflib.get(model_name)
    _compare_models(model_tflib, model_grpc, test_items)
    model_grpc.grpc_stub = None
    _compare_models(model_rest, model_grpc, test_items)
    assert model_grpc.grpc_stub
    lib_serving_rest.close()
    lib_serving_grpc.close()
