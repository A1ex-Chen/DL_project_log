def generate_mesh_t0(self, c_t=None, data=None, stats_dict={}):
    """ Generates mesh at first time step.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        """
    mesh = self.onet_generator.generate_from_latent(c_t, stats_dict=stats_dict)
    return mesh
