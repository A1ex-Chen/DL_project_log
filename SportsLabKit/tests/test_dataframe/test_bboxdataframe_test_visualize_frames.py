def test_visualize_frames(self):
    """Test for visualizing frames"""
    bbdf = BBoxDataFrame.from_dict({'home': {'0': {(0): [10, 10, 25, 25, 1]
        }, '1': {(0): [10, 10, 25, 25, 1]}}}, attributes=['bb_left',
        'bb_top', 'bb_width', 'bb_height', 'conf'])
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'video.avi')
        bbdf.visualize_frames(rgb_video_path, save_path)
        self.assertTrue(os.path.exists(save_path))
        cam = Camera(save_path)
        frame = cam.get_frame(0)
        print(frame)
        self.assertTrue(np.all(frame[0, 0] == [252, 0, 0]))
