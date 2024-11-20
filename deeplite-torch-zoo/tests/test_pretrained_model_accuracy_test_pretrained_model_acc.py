@pytest.mark.parametrize(('model_name', 'dataset_name',
    'reference_accuracy'), [('vgg19', 'tinyimagenet', 0.728), (
    'mobilenet_v2', 'tinyimagenet', 0.68), ('resnet18', 'tinyimagenet', 
    0.663), ('resnet34', 'tinyimagenet', 0.686), ('resnet50',
    'tinyimagenet', 0.73), ('mobilenetv3_large', 'vww', 0.891), (
    'mobilenetv3_small', 'vww', 0.892)])
@pytest.mark.slow
def test_pretrained_model_acc(model_name, dataset_name, reference_accuracy,
    abs_tolerance=0.05):
    model = get_model(model_name=model_name, dataset_name=dataset_name)
    test_loader = get_dataloaders(data_root=DATASET_PATHS[dataset_name],
        dataset_name=dataset_name, batch_size=BATCH_SIZE)[TEST_LOADER_KEY_MAP
        [dataset_name]]
    eval_fn = get_eval_function(model_name=model_name, dataset_name=
        dataset_name)
    accuracy = eval_fn(model, test_loader)[ACC_KEY_MAP[dataset_name]]
    assert pytest.approx(accuracy, abs_tolerance) == reference_accuracy
