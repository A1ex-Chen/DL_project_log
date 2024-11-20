def generate_from_latent(self, c_i, c_p=None, stats_dict={}, **kwargs):
    """ Generates mesh from latent.

        Args:
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
    threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
    t0 = time.time()
    box_size = 1 + self.padding
    if self.upsampling_steps == 0:
        nx = self.resolution0
        pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)
        values = self.eval_points(pointsf, c_i, c_p, **kwargs).cpu().numpy()
        value_grid = values.reshape(nx, nx, nx)
    else:
        mesh_extractor = MISE(self.resolution0, self.upsampling_steps,
            threshold)
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            pointsf = torch.FloatTensor(points).to(self.device)
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = box_size * (pointsf - 0.5)
            values = self.eval_points(pointsf, c_i, c_p, **kwargs).cpu().numpy(
                )
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()
        value_grid = mesh_extractor.to_dense()
    stats_dict['time (eval points)'] = time.time() - t0
    mesh = self.extract_mesh(value_grid, c_i, c_p, stats_dict=stats_dict)
    return mesh
