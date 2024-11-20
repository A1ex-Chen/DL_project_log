def create_loss(loc_loss_ftor, cls_loss_ftor, box_preds, cls_preds,
    cls_targets, cls_weights, reg_targets, reg_weights, num_class,
    encode_background_as_zeros=True, encode_rad_error_by_sin=False,
    box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(cls_targets, depth=num_class + 1,
        dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    else:
        reg_targets = limit_reg_target(reg_targets)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)
    return loc_losses, cls_losses
