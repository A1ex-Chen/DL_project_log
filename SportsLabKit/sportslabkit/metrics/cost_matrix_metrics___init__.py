def __init__(self, use_pred_pt=False, im_shape: tuple[float, float]=(1080, 
    1920)):
    self.normalizer = np.sqrt(im_shape[0] ** 2 + im_shape[1] ** 2)
    self.use_pred_pt = use_pred_pt
