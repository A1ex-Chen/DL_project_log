@staticmethod
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, inplace=False):
    if inplace:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br_hw = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            br_hw.sub_(tl)
            br_hw.clamp_min_(0)
            del tl
            area_ious = torch.prod(br_hw, 2)
            del br_hw
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 
                2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
            br_hw = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] /
                2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
            br_hw.sub_(tl)
            br_hw.clamp_min_(0)
            del tl
            area_ious = torch.prod(br_hw, 2)
            del br_hw
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        union = area_a[:, None] + area_b - area_ious
        area_ious.div_(union)
        return area_ious
    else:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 
                2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
            br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 
                2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        hw = (br - tl).clamp(min=0)
        area_i = torch.prod(hw, 2)
        ious = area_i / (area_a[:, None] + area_b - area_i)
        return ious
