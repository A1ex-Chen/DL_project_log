def load_field(self, input_idx_img=None):
    if input_idx_img is not None:
        idx_img = input_idx_img
    else:
        idx_img = 0
    data = {}
    if not self.mode == 'render':
        self.load_image(idx_img, data)
        if self.ref_img:
            self.load_ref_img(idx_img, data)
        if self.with_depth:
            self.load_depth(idx_img, data)
        if self.dpt_depth is not None:
            self.load_DPT_depth(idx_img, data)
    if self.with_camera:
        self.load_camera(idx_img, data)
    return data
