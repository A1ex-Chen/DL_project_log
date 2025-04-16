@staticmethod
def inside(bboxes: Tensor, left: float, top: float, right: float, bottom: float
    ) ->Tensor:
    return (bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) * (bboxes[...,
        2] <= right) * (bboxes[..., 3] <= bottom)
