def build_datasets(self):
    logging.info('Building datasets...')
    self.build_processors()
    build_info = self.config.build_info
    storage_path = build_info.storage
    datasets = dict()
    if not os.path.exists(storage_path):
        warnings.warn('storage path {} does not exist.'.format(storage_path))
    dataset_cls = self.train_dataset_cls
    datasets['train'] = dataset_cls(vis_processor=self.vis_processors[
        'train'], text_processor=self.text_processors['train'], ann_paths=[
        os.path.join(storage_path, 'filter_cap.json')], vis_root=os.path.
        join(storage_path, 'image'))
    return datasets
