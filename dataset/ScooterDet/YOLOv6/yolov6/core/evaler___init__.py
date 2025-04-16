def __init__(self, data, batch_size=32, img_size=640, conf_thres=0.03,
    iou_thres=0.65, device='', half=True, save_dir='', shrink_size=640,
    infer_on_rect=False, verbose=False, do_coco_metric=True, do_pr_metric=
    False, plot_curve=True, plot_confusion_matrix=False, specific_shape=
    False, height=640, width=640):
    assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
    self.data = data
    self.batch_size = batch_size
    self.img_size = img_size
    self.conf_thres = conf_thres
    self.iou_thres = iou_thres
    self.device = device
    self.half = half
    self.save_dir = save_dir
    self.shrink_size = shrink_size
    self.infer_on_rect = infer_on_rect
    self.verbose = verbose
    self.do_coco_metric = do_coco_metric
    self.do_pr_metric = do_pr_metric
    self.plot_curve = plot_curve
    self.plot_confusion_matrix = plot_confusion_matrix
    self.specific_shape = specific_shape
    self.height = height
    self.width = width
