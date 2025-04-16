def build(self):
    """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
    self.build_processors()
    build_info = self.config.build_info
    ann_info = build_info.annotations
    vis_info = build_info.get(self.data_type)
    datasets = dict()
    for split in ann_info.keys():
        if split not in ['train', 'val', 'test']:
            continue
        is_train = split == 'train'
        vis_processor = self.vis_processors['train'
            ] if is_train else self.vis_processors['eval']
        text_processor = self.text_processors['train'
            ] if is_train else self.text_processors['eval']
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]
        abs_ann_paths = []
        for ann_path in ann_paths:
            if not os.path.isabs(ann_path):
                ann_path = utils.get_cache_path(ann_path)
            abs_ann_paths.append(ann_path)
        ann_paths = abs_ann_paths
        vis_path = os.path.join(vis_info.storage, split)
        if not os.path.isabs(vis_path):
            vis_path = utils.get_cache_path(vis_path)
        if not os.path.exists(vis_path):
            warnings.warn('storage path {} does not exist.'.format(vis_path))
        dataset_cls = (self.train_dataset_cls if is_train else self.
            eval_dataset_cls)
        datasets[split] = dataset_cls(vis_processor=vis_processor,
            text_processor=text_processor, ann_paths=ann_paths, vis_root=
            vis_path)
    return datasets
