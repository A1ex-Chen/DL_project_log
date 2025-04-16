def write_tblog(tblogger, epoch, results, lrs, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar('val/mAP@0.5', results[0], epoch + 1)
    tblogger.add_scalar('val/mAP@0.50:0.95', results[1], epoch + 1)
    tblogger.add_scalar('train/iou_loss', losses[0], epoch + 1)
    tblogger.add_scalar('train/dist_focalloss', losses[1], epoch + 1)
    tblogger.add_scalar('train/cls_loss', losses[2], epoch + 1)
    tblogger.add_scalar('x/lr0', lrs[0], epoch + 1)
    tblogger.add_scalar('x/lr1', lrs[1], epoch + 1)
    tblogger.add_scalar('x/lr2', lrs[2], epoch + 1)
