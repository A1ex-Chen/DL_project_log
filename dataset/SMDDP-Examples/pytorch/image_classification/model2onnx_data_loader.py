def data_loader():
    loader, _ = get_syntetic_loader(None, image_size, bs, num_classes, False)
    for inp, _ in loader:
        yield inp
        break
