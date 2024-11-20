@mock.patch('deeplite_torch_zoo.api.datasets.classification.cifar._get_cifar')
def test_unit_cifar(*args):
    get_cifar100()
    get_cifar10()
