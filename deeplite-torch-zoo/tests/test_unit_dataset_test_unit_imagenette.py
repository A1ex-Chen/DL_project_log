@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.imagenette.PIL.Image')
@mock.patch('deeplite_torch_zoo.src.classification.datasets.imagenette.Path',
    MockedPath)
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.imagenette.download_and_extract_archive'
    )
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.imagenette.verify_str_arg')
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagenette.create_loader',
    create_loader)
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.imagenette.os.scandir')
def test_unit_imagenette(*args):
    MockedPath.rval = True
    get_imagenette_160('')
    imagenette = get_imagenette_320('')['train']
    imagenette._image_files = list(range(10))
    imagenette._labels = list(range(10))
    imagenette.transform = mock.MagicMock(side_effect=lambda x: x)
    imagenette.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = imagenette[0]
    assert label == 0
