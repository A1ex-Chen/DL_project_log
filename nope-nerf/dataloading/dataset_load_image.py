def load_image(self, idx, data={}):
    image = self.imgs[idx]
    data[None] = image
    if self.use_DPT:
        data_in = {'image': np.transpose(image, (1, 2, 0))}
        data_in = self.transform(data_in)
        data['normalised_img'] = data_in['image']
    data['idx'] = idx
