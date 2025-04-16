@pytest.mark.skipif(has_tensorflow, reason=
    'This test needs not Tensorflow to be run')
@pytest.mark.parametrize('ModelType', [AsyncTensorflowModel, TensorflowModel])
def test_tf_model_with_mixin_mro(ModelType):
    with pytest.raises(ImportError):
        ModelType()
