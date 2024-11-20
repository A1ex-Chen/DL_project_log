def build(self):
    self.build_processors()
    build_info = self.config.build_info
    datasets = dict()
    split = 'train'
    dataset_cls = self.train_dataset_cls
    datasets[split] = dataset_cls(vis_processor=self.vis_processors[split],
        text_processor=self.text_processors[split], location=build_info.storage
        ).inner_dataset
    return datasets
