@mock.patch('deeplite_torch_zoo.src.classification.datasets.flowers102.Path',
    MockedPath)
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.flowers102.PIL.Image')
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.flowers102.check_integrity'
    , return_value=True)
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.flowers102.download_and_extract_archive'
    )
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.flowers102.download_url')
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.flowers102.create_loader',
    create_loader)
@mock.patch('scipy.io.loadmat')
def test_unit_flowers102(*args):
    MockedPath.rval = False
    with pytest.raises(RuntimeError):
        flower102 = get_flowers102()
    MockedPath.rval = True
    flower102 = get_flowers102()['train']
    flower102._image_files = list(range(10))
    flower102._labels = list(range(10))
    flower102.transform = mock.MagicMock(side_effect=lambda x: x)
    flower102.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = flower102[0]
    assert label == 0
