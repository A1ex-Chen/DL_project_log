def _sbc(epoch):
    return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
