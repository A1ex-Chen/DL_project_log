@staticmethod
def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float
    ) ->Tensor:
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
    return bboxes
