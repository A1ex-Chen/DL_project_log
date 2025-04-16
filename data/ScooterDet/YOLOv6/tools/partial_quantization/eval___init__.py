def __init__(self, eval_cfg):
    task = eval_cfg['task']
    save_dir = eval_cfg['save_dir']
    half = eval_cfg['half']
    data = eval_cfg['data']
    batch_size = eval_cfg['batch_size']
    img_size = eval_cfg['img_size']
    device = eval_cfg['device']
    dataloader = None
    Evaler.check_task(task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    conf_thres = 0.03
    iou_thres = 0.65
    device = Evaler.reload_device(device, None, task)
    data = Evaler.reload_dataset(data) if isinstance(data, str) else data
    val = Evaler(data, batch_size, img_size, conf_thres, iou_thres, device,
        half, save_dir)
    val.stride = eval_cfg['stride']
    dataloader = val.init_data(dataloader, task)
    self.eval_cfg = eval_cfg
    self.half = half
    self.device = device
    self.task = task
    self.val = val
    self.val_loader = dataloader
