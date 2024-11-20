def get_json(self, save=False, verbose=False):
    """Return dataset JSON for Ultralytics HUB."""

    def _round(labels):
        """Update labels to integer class and 4 decimal place floats."""
        if self.task == 'detect':
            coordinates = labels['bboxes']
        elif self.task in {'segment', 'obb'}:
            coordinates = [x.flatten() for x in labels['segments']]
        elif self.task == 'pose':
            n, nk, nd = labels['keypoints'].shape
            coordinates = np.concatenate((labels['bboxes'], labels[
                'keypoints'].reshape(n, nk * nd)), 1)
        else:
            raise ValueError(f'Undefined dataset task={self.task}.')
        zipped = zip(labels['cls'], coordinates)
        return [[int(c[0]), *(round(float(x), 4) for x in points)] for c,
            points in zipped]
    for split in ('train', 'val', 'create_self_data'):
        self.stats[split] = None
        path = self.data.get(split)
        if path is None:
            continue
        files = [f for f in Path(path).rglob('*.*') if f.suffix[1:].lower() in
            IMG_FORMATS]
        if not files:
            continue
        if self.task == 'classify':
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(self.data[split])
            x = np.zeros(len(dataset.classes)).astype(int)
            for im in dataset.imgs:
                x[im[1]] += 1
            self.stats[split] = {'instance_stats': {'total': len(dataset),
                'per_class': x.tolist()}, 'image_stats': {'total': len(
                dataset), 'unlabelled': 0, 'per_class': x.tolist()},
                'labels': [{Path(k).name: v} for k, v in dataset.imgs]}
        else:
            from ultralytics.data import YOLODataset
            dataset = YOLODataset(img_path=self.data[split], data=self.data,
                task=self.task)
            x = np.array([np.bincount(label['cls'].astype(int).flatten(),
                minlength=self.data['nc']) for label in TQDM(dataset.labels,
                total=len(dataset), desc='Statistics')])
            self.stats[split] = {'instance_stats': {'total': int(x.sum()),
                'per_class': x.sum(0).tolist()}, 'image_stats': {'total':
                len(dataset), 'unlabelled': int(np.all(x == 0, 1).sum()),
                'per_class': (x > 0).sum(0).tolist()}, 'labels': [{Path(k).
                name: _round(v)} for k, v in zip(dataset.im_files, dataset.
                labels)]}
    if save:
        self.hub_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.hub_dir / 'stats.json'
        LOGGER.info(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)
    if verbose:
        LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
    return self.stats
