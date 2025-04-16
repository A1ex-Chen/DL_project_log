@mock.patch('deeplite_torch_zoo.src.classification.datasets.food101.json')
@mock.patch('deeplite_torch_zoo.src.classification.datasets.food101.Path',
    MockedPath)
@mock.patch('deeplite_torch_zoo.src.classification.datasets.food101.PIL.Image')
@mock.patch(
    'deeplite_torch_zoo.src.classification.datasets.food101.download_and_extract_archive'
    )
@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.food101.create_loader',
    create_loader)
@mock.patch('builtins.open', new_callable=mock_open, read_data='data')
def test_unit_food101(*args):
    MockedPath.rval = False
    with pytest.raises(RuntimeError):
        food101 = get_food101()
    MockedPath.rval = True
    food101 = get_food101()['train']
    food101._image_files = list(range(10))
    food101._labels = list(range(10))
    food101.transform = mock.MagicMock(side_effect=lambda x: x)
    food101.target_transform = mock.MagicMock(side_effect=lambda x: x)
    _, label = food101[0]
    assert label == 0
