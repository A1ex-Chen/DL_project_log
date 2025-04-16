def retrieve(self):
    if self.ice_assigned_ids is not None:
        if len(self.ice_assigned_ids) != self.ice_num:
            raise ValueError(
                "The number of 'ice_assigned_ids' is mismatched with 'ice_num'"
                )
        idx_list = self.ice_assigned_ids
    else:
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        idx_list = np.random.choice(num_idx, self.ice_num, replace=False
            ).tolist()
    rtr_idx_list = [idx_list for _ in trange(len(self.test_ds))]
    return rtr_idx_list
