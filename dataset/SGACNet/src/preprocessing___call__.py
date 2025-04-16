def __call__(self, sample):
    label = sample['label']
    h, w = label.shape
    sample['label_down'] = dict()
    for rate in self.downsampling_rates:
        label_down = cv2.resize(label.numpy(), (w // rate, h // rate),
            interpolation=cv2.INTER_NEAREST)
        sample['label_down'][rate] = torch.from_numpy(label_down)
    return sample
