def get_dataloader(image_size, bs, num_classes):
    """return dataloader for inference"""
    from image_classification.dataloaders import get_syntetic_loader

    def data_loader():
        loader, _ = get_syntetic_loader(None, image_size, bs, num_classes, 
            False)
        for inp, _ in loader:
            yield inp
            break
    return data_loader()
