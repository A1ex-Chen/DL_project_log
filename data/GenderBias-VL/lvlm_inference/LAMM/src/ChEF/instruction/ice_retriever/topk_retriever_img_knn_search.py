def knn_search(self, ice_num):
    res_list = self.forward(self.dataloader, process_bar=True, information=
        'Embedding test set...')
    rtr_idx_list = [[] for _ in range(len(res_list))]
    for entry in tqdm.tqdm(res_list):
        idx = entry['metadata']['id']
        embed = np.expand_dims(entry['embed'], axis=0)
        near_ids = self.index.search(embed, ice_num + 1)[1][0].tolist()
        near_ids = self.process_list(near_ids, idx, ice_num)
        rtr_idx_list[idx] = near_ids
    return rtr_idx_list
