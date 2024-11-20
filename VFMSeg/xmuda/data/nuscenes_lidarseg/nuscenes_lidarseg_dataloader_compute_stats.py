def compute_stats():
    preprocess_dir = 'path/to/data/nuscenes_lidarseg_preprocess/preprocess'
    nuscenes_dir = 'path/to/data/nuscenes'
    outdir = 'path/to/data/nuscenes_lidarseg_preprocess/stats'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = ('train_day', 'test_day', 'train_night', 'val_night',
        'test_night', 'train_usa', 'test_usa', 'train_singapore',
        'val_singapore', 'test_singapore')
    for split in splits:
        dataset = NuScenesLidarSegSCN(split=(split,), preprocess_dir=
            preprocess_dir, nuscenes_dir=nuscenes_dir)
        num_classes = len(dataset.class_names)
        points_per_class = np.zeros(num_classes, int)
        for i, data in enumerate(dataset.data):
            print('{}/{}'.format(i, len(dataset)))
            points_per_class += np.bincount(data['seg_labels'], minlength=
                num_classes)
        plt.barh(dataset.class_names, points_per_class)
        plt.grid(axis='x')
        for i, value in enumerate(points_per_class):
            x_pos = value
            y_pos = i
            if dataset.class_names[i] == 'driveable_surface':
                x_pos -= 0.25 * points_per_class.max()
                y_pos += 0.75
            plt.text(x_pos + 0.02 * points_per_class.max(), y_pos - 0.25,
                f'{value:,}', color='blue', fontweight='bold')
        plt.title(split)
        plt.tight_layout()
        plt.savefig(outdir / f'{split}.png')
        plt.close()
