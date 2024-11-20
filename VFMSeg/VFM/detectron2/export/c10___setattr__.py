def __setattr__(self, name, val):
    if name in ['im_info', 'indices', 'batch_extra_fields', 'image_size']:
        super().__setattr__(name, val)
    else:
        self.set(name, val)
