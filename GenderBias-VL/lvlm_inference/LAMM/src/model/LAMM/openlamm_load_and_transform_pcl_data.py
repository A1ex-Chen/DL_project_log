def load_and_transform_pcl_data(self, pcl_paths, device):
    if pcl_paths is None:
        return None
    pcl_output = []
    for pcl_path in pcl_paths:
        mesh_vertices = np.load(pcl_path)
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(
                height, 1)], 1)
        point_cloud, _ = random_sampling(point_cloud, self.num_points,
            return_choices=True)
        pcl_output.append(torch.from_numpy(point_cloud))
    return torch.stack(pcl_output, dim=0).to(device)
