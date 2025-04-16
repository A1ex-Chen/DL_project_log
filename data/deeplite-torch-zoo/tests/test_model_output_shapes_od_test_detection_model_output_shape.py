@pytest.mark.parametrize(('model_name', 'dataset_name', 'dataloader_kwargs',
    'output_shapes', 'download_checkpoint', 'check_shape'),
    DETECTION_MODEL_TESTS)
def test_detection_model_output_shape(model_name, dataset_name,
    dataloader_kwargs, output_shapes, download_checkpoint, check_shape):
    model = get_model(model_name=model_name, dataset_name='coco',
        pretrained=download_checkpoint)
    dataloader = get_dataloaders(data_root='./', dataset_name=dataset_name,
        batch_size=TEST_BATCH_SIZE, num_workers=0, **dataloader_kwargs)['test']
    img, *_ = next(iter(dataloader))
    img = img / 255
    y = model(img)
    y[0].sum().backward()
    if check_shape:
        assert y[0].shape == (4, *output_shapes[0], COCO_NUM_CLASSES + 5)
        assert y[1].shape == (4, *output_shapes[1], COCO_NUM_CLASSES + 5)
        assert y[2].shape == (4, *output_shapes[2], COCO_NUM_CLASSES + 5)
