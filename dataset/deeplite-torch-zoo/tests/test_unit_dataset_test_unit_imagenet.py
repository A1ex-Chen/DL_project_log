@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagenet.create_dataset',
    MockedPath)
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagenet.create_loader',
    create_loader)
def test_unit_imagenet(*args):
    get_imagenet('')
