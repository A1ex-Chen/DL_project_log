@property
def num_anchors_per_localization(self):
    num_rot = len(self._rotations)
    num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
    return num_rot * num_size
