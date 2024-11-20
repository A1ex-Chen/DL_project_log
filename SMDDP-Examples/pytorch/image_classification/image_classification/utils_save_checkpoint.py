def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
    checkpoint_dir='./', backup_filename=None):
    if not dist.is_initialized() or dist.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        print('SAVING {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir,
                'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir,
                backup_filename))
