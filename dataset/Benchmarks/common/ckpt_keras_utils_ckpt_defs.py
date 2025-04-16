def ckpt_defs(defs):
    new_defs = [{'name': 'ckpt_restart_mode', 'type': str, 'default':
        'auto', 'choices': ['off', 'auto', 'required'], 'help':
        'Mode to restart from a saved checkpoint file'}, {'name':
        'ckpt_checksum', 'type': str2bool, 'default': False, 'help':
        'Checksum the restart file after read+write'}, {'name':
        'ckpt_skip_epochs', 'type': int, 'default': 0, 'help':
        'Number of epochs to skip before saving epochs'}, {'name':
        'ckpt_directory', 'type': str, 'default': './save', 'help':
        'Base directory in which to save checkpoints'}, {'name':
        'ckpt_save_best', 'type': str2bool, 'default': True, 'help':
        'Toggle saving best model'}, {'name': 'ckpt_save_best_metric',
        'type': str, 'default': 'val_loss', 'help':
        'Metric for determining when to save best model'}, {'name':
        'ckpt_save_weights_only', 'type': str2bool, 'default': False,
        'help': 'Toggle saving only weights (not optimizer) (NYI)'}, {
        'name': 'ckpt_save_interval', 'type': int, 'default': 1, 'help':
        'Interval to save checkpoints'}, {'name': 'ckpt_keep_mode',
        'choices': ['linear', 'exponential'], 'help': 
        'Checkpoint saving mode. ' +
        "choices are 'linear' or 'exponential' "}, {'name':
        'ckpt_keep_limit', 'type': int, 'default': 1000000, 'help':
        'Limit checkpoints to keep'}]
    defs = defs + new_defs
    return defs
