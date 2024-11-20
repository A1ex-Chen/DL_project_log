def process_images(self):
    for split in ('train', 'val', 'test'):
        if self.data.get(split) is None:
            continue
        dataset = LoadImagesAndLabels(self.data[split])
        desc = f'{split} images'
        for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.
            im_files), total=dataset.n, desc=desc):
            pass
    print(f'Done. All images saved to {self.im_dir}')
    return self.im_dir
