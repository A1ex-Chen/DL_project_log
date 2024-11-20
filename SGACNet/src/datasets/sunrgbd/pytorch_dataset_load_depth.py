def load_depth(self, idx):
    if self.camera is None:
        depth_dir = self.depth_dir[self._split]['list']
    else:
        depth_dir = self.depth_dir[self._split]['dict'][self.camera]
    if self._depth_mode == 'raw':
        depth_file = depth_dir[idx].replace('depth_bfx', 'depth')
    else:
        depth_file = depth_dir[idx]
    fp = os.path.join(self._data_dir, depth_file)
    depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    return depth
