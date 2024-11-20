@pytest.fixture
def cifar100_dataloaders(data_root='./', batch_size=32):
    p = Path(data_root)
    dataloaders = get_dataloaders(dataset_name='cifar100', data_root=
        data_root, batch_size=batch_size, device='cpu')
    yield dataloaders
    (p / 'cifar-100-python.tar.gz').unlink()
    shutil.rmtree(p / 'cifar-100-python')
