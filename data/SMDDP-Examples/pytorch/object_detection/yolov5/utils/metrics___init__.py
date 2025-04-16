def __init__(self, nc, conf=0.25, iou_thres=0.45):
    self.matrix = np.zeros((nc + 1, nc + 1))
    self.nc = nc
    self.conf = conf
    self.iou_thres = iou_thres
