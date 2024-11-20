@mock.patch(
    'deeplite_torch_zoo.api.datasets.classification.vww.VisualWakeWordsClassification'
    )
@mock.patch('deeplite_torch_zoo.api.datasets.classification.vww.create_loader',
    create_loader)
def test_unit_vww(*args):
    get_vww('')
