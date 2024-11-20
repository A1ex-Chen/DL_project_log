@pytest.fixture
def dummy_tf_models():


    class DummyTFModel(TensorflowModel):
        CONFIGURATIONS = {'dummy_tf_model': {'asset': 'dummy_tf_model:0.0',
            'model_settings': {'output_dtypes': {'lambda': np.float32},
            'output_tensor_mapping': {'lambda': 'nothing'}, 'output_shapes':
            {'lambda': (3, 2, 1)}}}}

        def _is_empty(self, item):
            if item['input_1'][0, 0, 0] == -1:
                return True
            return False


    class DummyTFModelNoIsEmpty(TensorflowModel):
        CONFIGURATIONS = {'dummy_tf_model_no_is_empty': {'asset':
            'dummy_tf_model:0.0', 'model_settings': {'output_dtypes': {
            'lambda': np.float32}, 'output_tensor_mapping': {'lambda':
            'nothing'}, 'output_shapes': {'lambda': (3, 2, 1)},
            'tf_model_name': 'dummy_tf_model'}}}


    class DummyTFModelAsync(AsyncTensorflowModel):
        CONFIGURATIONS = {'dummy_tf_model_async': {'asset':
            'dummy_tf_model:0.0', 'model_settings': {'output_dtypes': {
            'lambda': np.float32}, 'output_tensor_mapping': {'lambda':
            'nothing'}, 'output_shapes': {'lambda': (3, 2, 1)},
            'tf_model_name': 'dummy_tf_model'}}}

        def _is_empty(self, item):
            if item['input_1'][0, 0, 0] == -1:
                return True
            return False
    return DummyTFModel, DummyTFModelAsync, DummyTFModelNoIsEmpty
