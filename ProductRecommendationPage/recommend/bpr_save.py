def save(self, factor, finished):
    save_path = self.save_path + '/model/'
    if not finished:
        save_path += str(factor) + '/'
    ensure_dir(save_path)
    logger.info('saving factors in {}'.format(save_path))
    item_bias = {iid: self.item_bias[self.i_inx[iid]] for iid in self.i_inx
        .keys()}
    uf = pd.DataFrame(self.user_factors, index=self.user_ids)
    it_f = pd.DataFrame(self.item_factors, index=self.item_ids)
    with open(save_path + 'user_factors.json', 'w') as outfile:
        outfile.write(uf.to_json())
    with open(save_path + 'item_factors.json', 'w') as outfile:
        outfile.write(it_f.to_json())
    with open(save_path + 'item_bias.data', 'wb') as ub_file:
        pickle.dump(item_bias, ub_file)
