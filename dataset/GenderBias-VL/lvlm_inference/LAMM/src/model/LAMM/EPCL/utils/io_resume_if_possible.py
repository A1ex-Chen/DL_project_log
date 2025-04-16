def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics
    last_checkpoint = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if not os.path.isfile(last_checkpoint):
        return epoch, best_val_metrics
    sd = torch.load(last_checkpoint, map_location=torch.device('cpu'))
    epoch = sd['epoch']
    best_val_metrics = sd['best_val_metrics']
    print(f'Found checkpoint at {epoch}. Resuming.')
    model_no_ddp.load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['optimizer'])
    print(
        f'Loaded model and optimizer state at {epoch}. Loaded best val metrics so far.'
        )
    return epoch, best_val_metrics
