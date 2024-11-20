def _inference(self, x):
    """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
    shape = x[0].shape
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in
            make_anchors(x, self.stride, 0.5))
        self.shape = shape
    if self.export and self.format in {'saved_model', 'pb', 'tflite',
        'edgetpu', 'tfjs'}:
        box = x_cat[:, :self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4:]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    if self.export and self.format in {'tflite', 'edgetpu'}:
        grid_h = shape[2]
        grid_w = shape[3]
        grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=
            box.device).reshape(1, 4, 1)
        norm = self.strides / (self.stride[0] * grid_size)
        dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.
            unsqueeze(0) * norm[:, :2])
    else:
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)
            ) * self.strides
    return torch.cat((dbox, cls.sigmoid()), 1)
