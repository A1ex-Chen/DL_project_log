def generate_meshes_t(self, c_t=None, data=None, stats_dict={}):
    """ Generates meshes at time steps > 0.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        """
    t = self.get_time_steps()
    meshes = []
    for i, t_v in enumerate(t):
        stats_dict_i = {}
        mesh = self.onet_generator.generate_from_latent(c_t[0, i:i + 1],
            stats_dict=stats_dict_i)
        meshes.append(mesh)
        for k, v in stats_dict_i.items():
            stats_dict[k] += v
    return meshes
