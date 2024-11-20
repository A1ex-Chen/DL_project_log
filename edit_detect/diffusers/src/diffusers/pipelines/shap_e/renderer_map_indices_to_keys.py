def map_indices_to_keys(self, output):
    h_map = {'sdf': (0, 1), 'density_coarse': (1, 2), 'density_fine': (2, 3
        ), 'stf': (3, 6), 'nerf_coarse': (6, 9), 'nerf_fine': (9, 12)}
    mapped_output = {k: output[..., start:end] for k, (start, end) in h_map
        .items()}
    return mapped_output
