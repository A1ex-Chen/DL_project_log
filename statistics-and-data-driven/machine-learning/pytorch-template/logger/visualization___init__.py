def __init__(self, log_dir, logger, enabled):
    self.writer = None
    self.selected_module = ''
    if enabled:
        log_dir = str(log_dir)
        succeeded = False
        for module in ['torch.utils.tensorboard', 'tensorboardX']:
            try:
                self.writer = importlib.import_module(module).SummaryWriter(
                    log_dir)
                succeeded = True
                break
            except ImportError:
                succeeded = False
            self.selected_module = module
        if not succeeded:
            message = (
                "Warning: visualization (Tensorboard) is configured to use, but currently not installed on this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                )
            logger.warning(message)
    self.step = 0
    self.mode = ''
    self.tb_writer_ftns = {'add_scalar', 'add_scalars', 'add_image',
        'add_images', 'add_audio', 'add_text', 'add_histogram',
        'add_pr_curve', 'add_embedding'}
    self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
    self.timer = datetime.now()
