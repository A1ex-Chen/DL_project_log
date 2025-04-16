def load_depth(self, idx):
    depth = self._load(self._depth_dir, self._files[self._depth_dir][idx])
    if depth.dtype == 'float16':
        depth = depth.astype('float32')
        depth[depth > 300] = 0
    return depth
