def _forward(self, xin):
    outputs = []
    cls_preds = []
    bbox_preds = []
    obj_preds = []
    origin_preds = []
    org_xy_shifts = []
    xy_shifts = []
    expanded_strides = []
    center_ltrbes = []
    cls_xs = xin[0::2]
    reg_xs = xin[1::2]
    in_type = xin[0].type()
    h, w = reg_xs[0].shape[2:4]
    h *= self.stride[0]
    w *= self.stride[0]
    for k, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride,
        cls_xs, reg_xs)):
        cls_output = self.cls_preds[k](cls_x)
        reg_output = self.reg_preds[k](reg_x)
        obj_output = self.obj_preds[k](reg_x)
        if self.training:
            batch_size = cls_output.shape[0]
            hsize, wsize = cls_output.shape[-2:]
            size = hsize * wsize
            cls_output = cls_output.view(batch_size, -1, size).permute(0, 2, 1
                ).contiguous()
            reg_output = reg_output.view(batch_size, 4, size).permute(0, 2, 1
                ).contiguous()
            obj_output = obj_output.view(batch_size, 1, size).permute(0, 2, 1
                ).contiguous()
            if self.use_l1:
                origin_preds.append(reg_output.clone())
            output, grid, xy_shift, expanded_stride, center_ltrb = (self.
                get_output_and_grid(reg_output, hsize, wsize, k,
                stride_this_level, in_type))
            org_xy_shifts.append(grid)
            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            center_ltrbes.append(center_ltrb)
            cls_preds.append(cls_output)
            bbox_preds.append(output)
            obj_preds.append(obj_output)
        else:
            output = torch.cat([reg_output, obj_output.sigmoid(),
                cls_output.sigmoid()], 1)
            outputs.append(output)
    if self.training:
        bbox_preds = torch.cat(bbox_preds, 1)
        obj_preds = torch.cat(obj_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        org_xy_shifts = torch.cat(org_xy_shifts, 1)
        xy_shifts = torch.cat(xy_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        center_ltrbes = torch.cat(center_ltrbes, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        else:
            origin_preds = bbox_preds.new_zeros(1)
        whwh = torch.Tensor([[w, h, w, h]]).type_as(bbox_preds)
        return (bbox_preds, cls_preds, obj_preds, origin_preds,
            org_xy_shifts, xy_shifts, expanded_strides, center_ltrbes, whwh)
    else:
        return outputs
