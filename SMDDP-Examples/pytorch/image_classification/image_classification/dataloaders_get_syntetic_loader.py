def get_syntetic_loader(data_path, image_size, batch_size, num_classes,
    one_hot, interpolation=None, augmentation=None, start_epoch=0, workers=
    None, _worker_init_fn=None, memory_format=torch.contiguous_format):
    return SynteticDataLoader(batch_size, num_classes, 3, image_size,
        image_size, one_hot, memory_format=memory_format), -1
