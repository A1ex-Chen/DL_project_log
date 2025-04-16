def save_checkpoint(checkpoint_dir, model_no_ddp, optimizer, epoch, args,
    best_val_metrics, filename=None):
    if not is_primary():
        return
    if filename is None:
        filename = f'checkpoint_{epoch:04d}.pth'
    checkpoint_name = os.path.join(checkpoint_dir, filename)
    sd = {'model': model_no_ddp.state_dict(), 'optimizer': optimizer.
        state_dict(), 'epoch': epoch, 'args': args, 'best_val_metrics':
        best_val_metrics}
    torch.save(sd, checkpoint_name)
