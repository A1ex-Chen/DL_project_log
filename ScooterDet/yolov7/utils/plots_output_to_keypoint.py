def output_to_keypoint(output):
    targets = []
    for i, o in enumerate(output):
        kpts = o[:, 6:]
        o = o[:, :6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])),
                conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)
