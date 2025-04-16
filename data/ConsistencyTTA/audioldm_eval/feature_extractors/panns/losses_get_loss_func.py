def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
