@pytest.mark.parametrize(('model_name', 'dataset_name', 'input_resolution',
    'num_inp_channels', 'target_output_shape'), CLASSIFICATION_MODEL_TESTS)
def test_classification_model_output_shape_arbitrary_num_clases(model_name,
    dataset_name, input_resolution, num_inp_channels, target_output_shape,
    download_checkpoint=True):
    model = get_model(model_name=model_name, num_classes=TEST_NUM_CLASSES,
        dataset_name=dataset_name, pretrained=download_checkpoint and 
        model_name not in NO_PRETRAINED_WEIGHTS)
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels,
        input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
