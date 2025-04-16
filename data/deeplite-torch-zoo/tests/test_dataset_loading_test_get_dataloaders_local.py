@pytest.mark.parametrize(('dataset_name', 'data_root',
    'train_dataloader_len', 'test_dataloader_len'), [('vww', str(
    DATASETS_ROOT / 'vww'), 901, 63), ('imagenet', str(DATASETS_ROOT /
    'imagenet16'), 1408, 332), ('imagenet', str(DATASETS_ROOT /
    'imagenet10'), 3010, 118), ('imagenet', str(DATASETS_ROOT / 'imagenet'),
    10010, 391), ('coco', str(DATASETS_ROOT / 'coco'), 11829, 500)])
@pytest.mark.local
def test_get_dataloaders_local(dataset_name, data_root,
    train_dataloader_len, test_dataloader_len):
    dataloaders = get_dataloaders(data_root=data_root, dataset_name=
        dataset_name, batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)
    assert len(dataloaders['train']) == train_dataloader_len
    assert len(dataloaders['test']) == test_dataloader_len
