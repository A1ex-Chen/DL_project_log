@staticmethod
def c2_postprocess(im_info, rpn_rois, rpn_roi_probs, tensor_mode):
    proposals = InstancesList(im_info=im_info, indices=rpn_rois[:, 0],
        extra_fields={'proposal_boxes': Caffe2Boxes(rpn_rois),
        'objectness_logits': (torch.Tensor, rpn_roi_probs)})
    if not tensor_mode:
        proposals = InstancesList.to_d2_instances_list(proposals)
    else:
        proposals = [proposals]
    return proposals
