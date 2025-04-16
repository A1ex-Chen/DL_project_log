def test_visualize_frame(self):
    """Test for visualizing frame"""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25, 1]
        }}}, attributes=['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    cam = Camera(rgb_video_path)
    frame = cam.get_frame(1)
    new_frame = bbdf.visualize_frame(1, frame.copy())
    self.assertTrue(np.all(new_frame[0, 0] == [255, 0, 0]))
