def forward(self, x, bev=None):
    x0 = self.downsample0(self.block0(x[0]))
    x1 = self.block1(x[1])
    x2 = self.upsample2(self.block2(x[2]))
    x3 = self.upsample3(self.block3(x[3]))
    xx = self.output_after_concate_fuse3210(torch.cat([x0, x1, x2, x3], dim=1))
    box_preds = self.conv_box(xx)
    cls_preds = self.conv_cls(xx)
    box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
    cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
    ret_dict = {'box_preds': box_preds, 'cls_preds': cls_preds}
    if self._use_direction_classifier:
        dir_cls_preds = self.conv_dir_cls(xx)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict['dir_cls_preds'] = dir_cls_preds
    return ret_dict
