def miou_pytorch(cm, ignore_index=None):
    return iou_pytorch(cm=cm, ignore_index=ignore_index).mean()
