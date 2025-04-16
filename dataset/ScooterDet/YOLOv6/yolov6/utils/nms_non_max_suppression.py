def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
    classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    num_classes = prediction.shape[2] - 5
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, 
        torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'
    max_wh = 4096
    max_nms = 30000
    time_limit = 10.0
    multi_label &= num_classes > 1
    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
        ] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):
        x = x[pred_candidates[img_idx]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False
                ).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None],
                class_idx[:, None].float()), 1)
        else:
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) >
                conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        num_box = x.shape[0]
        if not num_box:
            continue
        elif num_box > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + class_offset, x[:, 4]
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)
        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]
        output[img_idx] = x[keep_box_idx]
        if time.time() - tik > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break
    return output
