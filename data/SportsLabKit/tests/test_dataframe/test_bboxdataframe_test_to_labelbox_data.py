def test_to_labelbox_data(self):
    """Test for converting to labelbox data"""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25, 1],
        (1): [0, 0, 20, 20, 1]}, '2': {(2): [2, 1, 25, 25, 1]}}},
        attributes=['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    MockDataRow = namedtuple('DataRow', ['uid'])
    mock_data_row = MockDataRow('test')
    schema_lookup = {'home_1': '3q4fhvwui45yt', 'home_2': 'sadfjdhjf1241'}
    with mock.patch('uuid.uuid4', return_value='test_value'):
        data = bbdf.to_labelbox_data(mock_data_row, schema_lookup)
    ans = [{'uuid': 'test_value', 'schemaId': '3q4fhvwui45yt', 'dataRow': {
        'id': 'test'}, 'segments': [{'keyframes': [{'frame': 1, 'bbox': {
        'top': 10, 'left': 10, 'height': 25, 'width': 25}}, {'frame': 2,
        'bbox': {'top': 0, 'left': 0, 'height': 20, 'width': 20}}]}]}, {
        'uuid': 'test_value', 'schemaId': 'sadfjdhjf1241', 'dataRow': {'id':
        'test'}, 'segments': [{'keyframes': [{'frame': 3, 'bbox': {'top': 1,
        'left': 2, 'height': 25, 'width': 25}}]}]}]
    self.assertListEqual(data, ans)
