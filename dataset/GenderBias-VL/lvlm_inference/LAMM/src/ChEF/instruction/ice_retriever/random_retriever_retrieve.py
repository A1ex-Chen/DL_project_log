def retrieve(self):
    np.random.seed(self.seed)
    num_idx = len(self.index_ds)
    rtr_idx_list = []
    for _ in trange(len(self.test_ds)):
        idx_list = np.random.choice(num_idx, self.ice_num, replace=False
            ).tolist()
        rtr_idx_list.append(idx_list)
    return rtr_idx_list
