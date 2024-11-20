def yolo_bbox2segment(im_dir, save_dir=None, sam_model='sam_b.pt'):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in YOLO format. Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from tqdm import tqdm
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils import LOGGER
    from ultralytics.utils.ops import xywh2xyxy
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]['segments']) > 0:
        LOGGER.info(
            'Segmentation labels detected, no need to generate new ones!')
        return
    LOGGER.info(
        'Detection labels detected, generating segment labels by SAM model!')
    sam_model = SAM(sam_model)
    for label in tqdm(dataset.labels, total=len(dataset.labels), desc=
        'Generating segment labels'):
        h, w = label['shape']
        boxes = label['bboxes']
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label['im_file'])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False,
            save=False)
        label['segments'] = sam_results[0].masks.xyn
    save_dir = Path(save_dir) if save_dir else Path(im_dir
        ).parent / 'labels-segment'
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label['im_file']).with_suffix('.txt').name
        txt_file = save_dir / lb_name
        cls = label['cls']
        for i, s in enumerate(label['segments']):
            line = int(cls[i]), *s.reshape(-1)
            texts.append(('%g ' * len(line)).rstrip() % line)
        if texts:
            with open(txt_file, 'a') as f:
                f.writelines(text + '\n' for text in texts)
    LOGGER.info(f'Generated segment labels saved in {save_dir}')
