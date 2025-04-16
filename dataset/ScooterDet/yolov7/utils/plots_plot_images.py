def plot_images(images, targets, paths=None, fname='images.jpg', names=None,
    max_size=640, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255
    tl = 3
    tf = max(tl - 1, 1)
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)
    colors = color_list()
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, img in enumerate(images):
        if i == max_subplots:
            break
        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))
        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6
            conf = None if labels else image_targets[:, 6]
            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j]
                        )
                    plot_one_box(box, mosaic, label=label, color=color,
                        line_thickness=tl)
        if paths:
            label = Path(paths[i]).name[:40]
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
                0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 
                5), 0, tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.
                LINE_AA)
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h
            ), (255, 255, 255), thickness=3)
    if fname:
        r = min(1280.0 / max(h, w) / ns, 1.0)
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)),
            interpolation=cv2.INTER_AREA)
        Image.fromarray(mosaic).save(fname)
    return mosaic
