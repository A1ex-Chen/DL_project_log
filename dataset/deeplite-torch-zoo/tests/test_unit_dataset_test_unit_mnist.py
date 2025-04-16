@mock.patch('deeplite_torch_zoo.api.datasets.classification.mnist.torchvision')
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.mnist.create_loader',
    create_loader)
def test_unit_mnist(*args):
    get_mnist('')
