def get_dummy_components(self, time_cond_proj_dim=None):
    return super().get_dummy_components('multi_adapter', time_cond_proj_dim
        =time_cond_proj_dim)
