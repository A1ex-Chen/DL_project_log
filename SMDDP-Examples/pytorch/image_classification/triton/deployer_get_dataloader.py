def get_dataloader(args):
    """ return dataloader for inference """
    from image_classification.dataloaders import get_syntetic_loader

    def data_loader():
        loader, _ = get_syntetic_loader(None, 128, 1000, True, fp16=args.fp16)
        processed = 0
        for inp, _ in loader:
            yield inp
            processed += 1
            if processed > 10:
                break
    return data_loader()
