@staticmethod
def to_center_base(bboxes: Tensor) ->Tensor:
    return torch.stack([(bboxes[..., 0] + bboxes[..., 2]) / 2, (bboxes[...,
        1] + bboxes[..., 3]) / 2, bboxes[..., 2] - bboxes[..., 0], bboxes[
        ..., 3] - bboxes[..., 1]], dim=-1)
