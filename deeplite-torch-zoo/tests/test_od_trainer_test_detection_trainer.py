@pytest.mark.parametrize(('model_name',), TEST_MODELS)
def test_detection_trainer(model_name):
    model = Detector(model_name=model_name)
    model.train(data='coco8.yaml', epochs=1)
