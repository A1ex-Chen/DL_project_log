def should_backup_checkpoint(args):

    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc
