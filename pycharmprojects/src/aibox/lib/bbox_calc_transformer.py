@staticmethod
def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) ->Tensor:
    center_based_src_bboxes = BBox.to_center_base(src_bboxes)
    center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
    transformers = torch.stack([(center_based_dst_bboxes[..., 0] -
        center_based_src_bboxes[..., 0]) / center_based_src_bboxes[..., 2],
        (center_based_dst_bboxes[..., 1] - center_based_src_bboxes[..., 1]) /
        center_based_src_bboxes[..., 3], torch.log(center_based_dst_bboxes[
        ..., 2] / center_based_src_bboxes[..., 2]), torch.log(
        center_based_dst_bboxes[..., 3] / center_based_src_bboxes[..., 3])],
        dim=-1)
    return transformers
