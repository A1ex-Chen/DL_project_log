def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    paths = self.batch_data[2]
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255),
            thickness=2)
        cv2.putText(mosaic, f'{os.path.basename(paths[i])[:40]}', (x + 5, y +
            15), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220),
            thickness=1)
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6
            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                box = [int(k) for k in box]
                cls = classes[j]
                color = tuple([int(x) for x in self.color[cls]])
                cls = self.data_dict['names'][cls] if self.data_dict['names'
                    ] else cls
                if labels:
                    label = f'{cls}'
                    cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]
                        ), color, thickness=1)
                    cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.
                        FONT_HERSHEY_COMPLEX, 0.5, color, thickness=1)
    self.vis_train_batch = mosaic.copy()
