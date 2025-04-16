def __init__(self, *args, data=None, use_segments=False, use_keypoints=
    False, **kwargs):
    self.use_segments = use_segments
    self.use_keypoints = use_keypoints
    self.data = data
    assert not (self.use_segments and self.use_keypoints
        ), 'Can not use both segments and keypoints.'
    super().__init__(*args, **kwargs)
