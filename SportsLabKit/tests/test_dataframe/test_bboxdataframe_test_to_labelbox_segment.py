def test_to_labelbox_segment(self):
    """Test for converting to labelbox segment"""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25, 1],
        (1): [0, 0, 20, 20, 1]}, '2': {(2): [2, 1, 25, 25, 1]}}},
        attributes=['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    data = bbdf.to_labelbox_segment()
    ans = {'home_1': [{'keyframes': [{'frame': 1, 'bbox': {'top': 10,
        'left': 10, 'height': 25, 'width': 25}}, {'frame': 2, 'bbox': {
        'top': 0, 'left': 0, 'height': 20, 'width': 20}}]}], 'home_2': [{
        'keyframes': [{'frame': 3, 'bbox': {'top': 1, 'left': 2, 'height': 
        25, 'width': 25}}]}]}
    self.assertDictEqual(data, ans)
