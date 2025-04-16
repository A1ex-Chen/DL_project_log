@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagewoof.Imagewoof')
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagewoof.create_loader',
    create_loader)
def test_unit_imagewoof(*args):
    get_imagewoof_160('')
    get_imagewoof_320('')
