@staticmethod
def get_lr_scheduler(args, cfg, optimizer):
    epochs = args.epochs
    lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
    return lr_scheduler, lf
