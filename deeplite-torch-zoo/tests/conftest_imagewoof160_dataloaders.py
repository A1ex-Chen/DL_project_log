@pytest.fixture
def imagewoof160_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(dataset_name='imagewoof_160', data_root=
        data_root, batch_size=batch_size, map_to_imagenet_labels=True,
        device='cpu')
    yield dataloaders
    (p / 'imagewoof160.zip').unlink()
    shutil.rmtree(p / 'imagewoof160')
