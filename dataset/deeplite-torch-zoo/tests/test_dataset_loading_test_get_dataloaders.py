@pytest.mark.parametrize(('dataset_name', 'tmp_dataset_files',
    'tmp_dataset_folders', 'train_dataloader_len', 'test_dataloader_len'),
    [('cifar100', ('cifar-100-python.tar.gz',), ('cifar-100-python',), 390,
    79), ('cifar10', ('cifar-10-python.tar.gz',), ('cifar-10-batches-py',),
    390, 79), ('imagenette', ('imagenette.zip',), ('imagenette',), 73, 31),
    ('imagewoof', ('imagewoof.zip',), ('imagewoof',), 70, 31), ('mnist', (),
    ('MNIST',), 468, 79), ('coco128', (), (), 1, 1)])
def test_get_dataloaders(dataset_name, tmp_dataset_files,
    tmp_dataset_folders, train_dataloader_len, test_dataloader_len,
    data_root='./'):
    p = Path(data_root)
    extra_loader_kwargs = {}
    if dataset_name not in ('coco128',):
        extra_loader_kwargs = {'test_batch_size': BATCH_SIZE}
    dataloaders = get_dataloaders(data_root=data_root, dataset_name=
        dataset_name, batch_size=BATCH_SIZE, **extra_loader_kwargs)
    assert len(dataloaders['train']) == train_dataloader_len
    assert len(dataloaders['test']) == test_dataloader_len
    for file in tmp_dataset_files:
        (p / file).unlink()
    for folder in tmp_dataset_folders:
        shutil.rmtree(p / folder)
