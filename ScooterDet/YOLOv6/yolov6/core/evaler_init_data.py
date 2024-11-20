def init_data(dataloader, task):
    self.is_coco = self.data.get('is_coco', False)
    self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range
        (1000))
    pad = 0.0
    dataloader = create_dataloader(self.data[task if task in ('train',
        'val', 'test') else 'val'], self.img_size, self.batch_size, self.
        stride, check_labels=True, pad=pad, rect=False, data_dict=self.data,
        task=task)[0]
    return dataloader
