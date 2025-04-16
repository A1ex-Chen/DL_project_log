def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou
