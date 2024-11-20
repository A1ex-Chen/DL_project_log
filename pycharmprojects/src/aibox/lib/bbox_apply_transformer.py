@staticmethod
def apply_transformer(src_bboxes: Tensor, transformers: Tensor) ->Tensor:
    center_based_src_bboxes = BBox.to_center_base(src_bboxes)
    center_based_dst_bboxes = torch.stack([transformers[..., 0] *
        center_based_src_bboxes[..., 2] + center_based_src_bboxes[..., 0], 
        transformers[..., 1] * center_based_src_bboxes[..., 3] +
        center_based_src_bboxes[..., 1], torch.exp(transformers[..., 2]) *
        center_based_src_bboxes[..., 2], torch.exp(transformers[..., 3]) *
        center_based_src_bboxes[..., 3]], dim=-1)
    dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
    return dst_bboxes
