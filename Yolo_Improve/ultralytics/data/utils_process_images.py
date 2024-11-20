def process_images(self):
    """Compress images for Ultralytics HUB."""
    from ultralytics.data import YOLODataset
    self.im_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'create_self_data'):
        if self.data.get(split) is None:
            continue
        dataset = YOLODataset(img_path=self.data[split], data=self.data)
        with ThreadPool(NUM_THREADS) as pool:
            for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total
                =len(dataset), desc=f'{split} images'):
                pass
    LOGGER.info(f'Done. All images saved to {self.im_dir}')
    return self.im_dir
