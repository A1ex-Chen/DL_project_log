def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
    classes=None, agnostic=False, multi_label=False, labels=(), max_det=300,
    nm=0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device
    mps = 'mps' in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(),
                mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) >
                conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1,
                keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if time.time() - t > time_limit:
            LOGGER.warning(
                f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break
    return output
