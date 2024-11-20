def __init__(self, dataset_name, distributed=True, output_dir=None, *,
    sem_seg_loading_fn=load_image_into_numpy_array, num_classes=None,
    ignore_label=None):
    super().__init__(dataset_name, distributed=True, output_dir=None,
        sem_seg_loading_fn=load_image_into_numpy_array, num_classes=None,
        ignore_label=None)
    meta = MetadataCatalog.get(dataset_name)
    self.label_group = meta.get('label_group', None)
    self.n_merged_cls = 58
