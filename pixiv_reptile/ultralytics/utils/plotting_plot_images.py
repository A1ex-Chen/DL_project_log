@threaded
def plot_images(images: Union[torch.Tensor, np.ndarray], batch_idx: Union[
    torch.Tensor, np.ndarray], cls: Union[torch.Tensor, np.ndarray], bboxes:
    Union[torch.Tensor, np.ndarray]=np.zeros(0, dtype=np.float32), confs:
    Optional[Union[torch.Tensor, np.ndarray]]=None, masks: Union[torch.
    Tensor, np.ndarray]=np.zeros(0, dtype=np.uint8), kpts: Union[torch.
    Tensor, np.ndarray]=np.zeros((0, 51), dtype=np.float32), paths:
    Optional[List[str]]=None, fname: str='images.jpg', names: Optional[Dict
    [int, str]]=None, on_plot: Optional[Callable]=None, max_size: int=1920,
    max_subplots: int=16, save: bool=True, conf_thres: float=0.25) ->Optional[
    np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    if np.max(images[0]) <= 1:
        images *= 255
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y:y + h, x:x + w, :] = images[i].transpose(1, 2, 0)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs,
        pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255),
            width=2)
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40],
                txt_color=(220, 220, 220))
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype('int')
            labels = confs is None
            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:
                        boxes[..., [0, 2]] *= w
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(
                    boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f'{c}' if labels else f'{c} {conf[j]:.1f}'
                        annotator.box_label(box, label, color=color,
                            rotated=is_obb)
            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f'{c}', txt_color=color,
                        box_style=True)
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max(
                        ) <= 1.01:
                        kpts_[..., 0] *= w
                        kpts_[..., 1] *= h
                    elif scale < 1:
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)
            if len(masks):
                if idx.shape[0] == masks.shape[0]:
                    image_masks = masks[idx]
                else:
                    image_masks = masks[[i]]
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x +
                                w, :][mask] * 0.4 + np.array(color) * 0.6
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)
