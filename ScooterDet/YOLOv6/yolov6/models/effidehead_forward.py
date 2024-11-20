def forward(self, x):
    if self.training:
        cls_score_list = []
        reg_distri_list = []
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            cls_output = torch.sigmoid(cls_output)
            cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)
        return x, cls_score_list, reg_distri_list
    else:
        cls_score_list = []
        reg_dist_list = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            if self.use_dfl:
                reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]
                    ).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
            cls_output = torch.sigmoid(cls_output)
            if self.export:
                cls_score_list.append(cls_output)
                reg_dist_list.append(reg_output)
            else:
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
        if self.export:
            return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(
                cls_score_list, reg_dist_list))
        cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
        reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
        anchor_points, stride_tensor = generate_anchors(x, self.stride,
            self.grid_cell_size, self.grid_cell_offset, device=x[0].device,
            is_eval=True, mode='af')
        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh'
            )
        pred_bboxes *= stride_tensor
        return torch.cat([pred_bboxes, torch.ones((b, pred_bboxes.shape[1],
            1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
            cls_score_list], axis=-1)
