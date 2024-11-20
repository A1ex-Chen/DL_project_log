def load_ref_img(self, idx, data={}):
    if self.random_ref:
        if idx == self.N_imgs - 1:
            ref_idx = idx - 1
        else:
            ran_idx = random.randint(1, min(self.random_ref, self.N_imgs -
                idx - 1))
            ref_idx = idx + ran_idx
    image = self.imgs[ref_idx]
    if self.dpt_depth is not None:
        dpt = self.dpt_depth[ref_idx]
        data['ref_dpts'] = dpt
    if self.use_DPT:
        data_in = {'image': np.transpose(image, (1, 2, 0))}
        data_in = self.transform(data_in)
        normalised_ref_img = data_in['image']
        data['normalised_ref_img'] = normalised_ref_img
    if self.with_depth:
        depth = self.depth[ref_idx]
        data['ref_depths'] = depth
    data['ref_imgs'] = image
    data['ref_idxs'] = ref_idx
