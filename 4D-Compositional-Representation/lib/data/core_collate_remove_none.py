def collate_remove_none(batch):
    """ Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    """
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)
