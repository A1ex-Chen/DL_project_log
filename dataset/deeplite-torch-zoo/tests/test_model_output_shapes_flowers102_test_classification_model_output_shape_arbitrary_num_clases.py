@pytest.mark.slow
@pytest.mark.parametrize(('model_name', 'dataset_name', 'input_resolution',
    'num_inp_channels', 'target_output_shape'), FLOWERS_MODEL_TESTS)
def test_classification_model_output_shape_arbitrary_num_clases(model_name,
    dataset_name, input_resolution, num_inp_channels, target_output_shape,
    download_checkpoint=False):
    model = get_model(model_name=model_name, num_classes=TEST_NUM_CLASSES,
        dataset_name=dataset_name, pretrained=download_checkpoint)
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels,
        input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
