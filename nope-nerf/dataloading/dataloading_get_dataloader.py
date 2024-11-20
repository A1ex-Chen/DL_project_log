def get_dataloader(cfg, mode='train', shuffle=True, n_views=None):
    """ Return dataloader instance

    Instansiate dataset class and dataloader and 
    return dataloader
    
    Args:
        cfg (dict): imported config for dataloading
        mode (str): tran/eval/render/all
        shuffle (bool): as name
        n_views (int): specify number of views during rendering
    """
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']
    fields = get_data_fields(cfg, mode)
    if n_views is not None and mode == 'render':
        n_views = n_views
    else:
        n_views = fields['img'].N_imgs
    dataset = OurDataset(fields, n_views=n_views, mode=mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=shuffle, pin_memory=True)
    return dataloader, fields
