def __init__(self, batch_size, batch_num, calib_img_dir, input_w, input_h):
    self.index = 0
    self.length = batch_num
    self.batch_size = batch_size
    self.input_h = input_h
    self.input_w = input_w
    self.img_list = [os.path.join(calib_img_dir, x) for x in os.listdir(
        calib_img_dir) if os.path.splitext(x)[-1] in IMG_FORMATS]
    assert len(self.img_list
        ) > self.batch_size * self.length, '{} must contains more than '.format(
        calib_img_dir) + str(self.batch_size * self.length
        ) + ' images to calib'
    print('found all {} images to calib.'.format(len(self.img_list)))
    self.calibration_data = np.zeros((self.batch_size, 3, input_h, input_w),
        dtype=np.float32)
