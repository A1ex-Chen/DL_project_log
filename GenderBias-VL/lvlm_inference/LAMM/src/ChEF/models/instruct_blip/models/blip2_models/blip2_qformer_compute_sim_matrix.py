def compute_sim_matrix(self, data_loader, task_cfg):
    """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
    k_test = task_cfg.k_test
    return compute_sim_matrix(model=self, data_loader=data_loader, k_test=
        k_test)
