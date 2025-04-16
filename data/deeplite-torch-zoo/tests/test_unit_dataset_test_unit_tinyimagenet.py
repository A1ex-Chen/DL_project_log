@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.tiny_imagenet.datasets')
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.tiny_imagenet.create_loader'
    , create_loader)
def test_unit_tinyimagenet(*args):
    get_tinyimagenet('')
