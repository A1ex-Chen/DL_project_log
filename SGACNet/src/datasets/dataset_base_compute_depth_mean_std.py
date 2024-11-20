def compute_depth_mean_std(self, force_recompute=False):
    assert self.split == 'train'
    depth_stats_filepath = os.path.join(self.source_path,
        f'depth_{self.depth_mode}_mean_std.pickle')
    if not force_recompute and os.path.exists(depth_stats_filepath):
        depth_stats = pickle.load(open(depth_stats_filepath, 'rb'))
        print(f'Loaded depth mean and std from {depth_stats_filepath}')
        print(depth_stats)
        return depth_stats
    print('Compute mean and std for depth images.')
    pixel_sum = np.float64(0)
    pixel_nr = np.uint64(0)
    std_sum = np.float64(0)
    print('Compute mean')
    for i in range(len(self)):
        depth = self.load_depth(i)
        if self.depth_mode == 'raw':
            depth_valid = depth[depth > 0]
        else:
            depth_valid = depth.flatten()
        pixel_sum += np.sum(depth_valid)
        pixel_nr += np.uint64(len(depth_valid))
        print(f'\r{i + 1}/{len(self)}', end='')
    print()
    mean = pixel_sum / pixel_nr
    print('Compute std')
    for i in range(len(self)):
        depth = self.load_depth(i)
        if self.depth_mode == 'raw':
            depth_valid = depth[depth > 0]
        else:
            depth_valid = depth.flatten()
        std_sum += np.sum(np.square(depth_valid - mean))
        print(f'\r{i + 1}/{len(self)}', end='')
    print()
    std = np.sqrt(std_sum / pixel_nr)
    depth_stats = {'mean': mean, 'std': std}
    print(depth_stats)
    with open(depth_stats_filepath, 'wb') as f:
        pickle.dump(depth_stats, f)
    return depth_stats
