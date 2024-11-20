def get_json(self, save=False, verbose=False):

    def _round(labels):
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in
            labels]
    for split in ('train', 'val', 'test'):
        if self.data.get(split) is None:
            self.stats[split] = None
            continue
        dataset = LoadImagesAndLabels(self.data[split])
        x = np.array([np.bincount(label[:, 0].astype(int), minlength=self.
            data['nc']) for label in tqdm(dataset.labels, total=dataset.n,
            desc='Statistics')])
        self.stats[split] = {'instance_stats': {'total': int(x.sum()),
            'per_class': x.sum(0).tolist()}, 'image_stats': {'total':
            dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
            'per_class': (x > 0).sum(0).tolist()}, 'labels': [{str(Path(k).
            name): _round(v.tolist())} for k, v in zip(dataset.im_files,
            dataset.labels)]}
    if save:
        stats_path = self.hub_dir / 'stats.json'
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)
    if verbose:
        print(json.dumps(self.stats, indent=2, sort_keys=False))
    return self.stats
