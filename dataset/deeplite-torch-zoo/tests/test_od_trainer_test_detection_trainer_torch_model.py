@pytest.mark.parametrize(('model_name',), TEST_MODELS)
def test_detection_trainer_torch_model(model_name):
    torch_model = get_model(model_name=model_name, dataset_name='coco',
        pretrained=False, custom_head='yolo8')
    model = Detector(torch_model=torch_model)
    model.train(data='coco8.yaml', epochs=1)
