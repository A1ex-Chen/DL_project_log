def generate_mesh(self, data, return_stats=True):
    """ Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
    self.model.eval()
    device = self.device
    stats_dict = {}
    inputs = data.get('inputs', torch.empty(1, 0)).to(device)
    kwargs = {}
    if self.preprocessor is not None:
        t0 = time.time()
        with torch.no_grad():
            inputs = self.preprocessor(inputs)
        stats_dict['time (preprocess)'] = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        c = self.model.encode_inputs(inputs)
    stats_dict['time (encode inputs)'] = time.time() - t0
    z = self.model.get_z_from_prior((1,), sample=self.sample).to(device)
    mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)
    if return_stats:
        return mesh, stats_dict
    else:
        return mesh
