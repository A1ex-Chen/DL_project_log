def forward(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training or self.no_post_processing:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in
            make_anchors(x, self.stride, 0.5))
        self.shape = shape
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.export and self.format in ('saved_model', 'pb', 'tflite',
        'edgetpu', 'tfjs'):
        box = x_cat[:, :self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4:]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1
        ) * self.strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)
