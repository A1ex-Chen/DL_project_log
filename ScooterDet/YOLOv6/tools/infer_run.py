@torch.no_grad()
def run(weights=osp.join(ROOT, 'yolov6s.pt'), source=osp.join(ROOT,
    'data/images'), webcam=False, webcam_addr=0, yaml=None, img_size=640,
    conf_thres=0.4, iou_thres=0.45, max_det=1000, device='', save_txt=False,
    not_save_img=False, save_dir=None, view_img=True, classes=None,
    agnostic_nms=False, project=osp.join(ROOT, 'runs/inference'), name=
    'exp', hide_labels=False, hide_conf=False, half=False):
    """ Inference process, supporting inference on one image file or directory which containing images.
    Args:
        weights: The path of model.pt, e.g. yolov6s.pt
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        not_save_img: Do not save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name, e.g. 'exp'
        line_thickness: Bounding box thickness (pixels), e.g. 3
        hide_labels: Hide labels, e.g. False
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference, e.g. False
    """
    if save_dir is None:
        save_dir = osp.join(project, name)
        save_txt_path = osp.join(save_dir, 'labels')
    else:
        save_txt_path = save_dir
    if (not not_save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning('Save directory already existed')
    if save_txt:
        save_txt_path = osp.join(save_dir, 'labels')
        if not osp.exists(save_txt_path):
            os.makedirs(save_txt_path)
    inferer = Inferer(source, webcam, webcam_addr, weights, device, yaml,
        img_size, half)
    inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det,
        save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)
    if save_txt or not not_save_img:
        LOGGER.info(f'Results saved to {save_dir}')
