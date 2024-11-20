@staticmethod
def from_center_base(center_based_bboxes: Tensor) ->Tensor:
    return torch.stack([center_based_bboxes[..., 0] - center_based_bboxes[
        ..., 2] / 2, center_based_bboxes[..., 1] - center_based_bboxes[...,
        3] / 2, center_based_bboxes[..., 0] + center_based_bboxes[..., 2] /
        2, center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2],
        dim=-1)
