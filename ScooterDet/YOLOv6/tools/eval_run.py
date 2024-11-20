@torch.no_grad()
def run(data, weights=None, batch_size=32, img_size=640, conf_thres=0.03,
    iou_thres=0.65, task='val', device='', half=False, model=None,
    dataloader=None, save_dir='', name='', shrink_size=640,
    letterbox_return_int=False, infer_on_rect=False, reproduce_640_eval=
    False, eval_config_file='./configs/experiment/eval_640_repro.py',
    verbose=False, do_coco_metric=True, do_pr_metric=False, plot_curve=
    False, plot_confusion_matrix=False, config_file=None, specific_shape=
    False, height=640, width=640):
    """ Run the evaluation process

    This function is the main process of evaluation, supporting image file and dir containing images.
    It has tasks of 'val', 'train' and 'speed'. Task 'train' processes the evaluation during training phase.
    Task 'val' processes the evaluation purely and return the mAP of model.pt. Task 'speed' processes the
    evaluation of inference speed of model.pt.

    """
    Evaler.check_task(task)
    if task == 'train':
        save_dir = save_dir
    else:
        save_dir = str(increment_name(osp.join(save_dir, name)))
        os.makedirs(save_dir, exist_ok=True)
    Evaler.check_thres(conf_thres, iou_thres, task)
    device = Evaler.reload_device(device, model, task)
    half = device.type != 'cpu' and half
    data = Evaler.reload_dataset(data, task) if isinstance(data, str) else data
    if specific_shape:
        height = check_img_size(height, 32, floor=256)
        width = check_img_size(width, 32, floor=256)
    else:
        img_size = check_img_size(img_size, 32, floor=256)
    val = Evaler(data, batch_size, img_size, conf_thres, iou_thres, device,
        half, save_dir, shrink_size, infer_on_rect, verbose, do_coco_metric,
        do_pr_metric, plot_curve, plot_confusion_matrix, specific_shape=
        specific_shape, height=height, width=width)
    model = val.init_model(model, weights, task)
    dataloader = val.init_data(dataloader, task)
    model.eval()
    pred_result, vis_outputs, vis_paths = val.predict_model(model,
        dataloader, task)
    eval_result = val.eval_model(pred_result, model, dataloader, task)
    return eval_result, vis_outputs, vis_paths
