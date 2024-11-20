def __call__(self, sample):
    image = np.transpose(sample['image'], (2, 0, 1))
    sample['image'] = np.ascontiguousarray(image).astype(np.float32)
    if 'mask' in sample:
        sample['mask'] = sample['mask'].astype(np.float32)
        sample['mask'] = np.ascontiguousarray(sample['mask'])
    if 'disparity' in sample:
        disparity = sample['disparity'].astype(np.float32)
        sample['disparity'] = np.ascontiguousarray(disparity)
    if 'depth' in sample:
        depth = sample['depth'].astype(np.float32)
        sample['depth'] = np.ascontiguousarray(depth)
    return sample
