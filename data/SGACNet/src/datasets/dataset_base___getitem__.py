def __getitem__(self, idx):
    sample = {'image': self.load_image(idx), 'depth': self.load_depth(idx),
        'label': self.load_label(idx)}
    if self.split != 'train':
        sample['label_orig'] = sample['label'].copy()
    if self.with_input_orig:
        sample['image_orig'] = sample['image'].copy()
        sample['depth_orig'] = sample['depth'].copy().astype('float32')
    sample = self.preprocessor(sample)
    return sample
