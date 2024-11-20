def __init__(self, box_format='xywh', iou_type='ciou', reduction='none',
    eps=1e-07):
    """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
    self.box_format = box_format
    self.iou_type = iou_type.lower()
    self.reduction = reduction
    self.eps = eps
