def forward(self, x):
    if self.training:
        device = x[0].device
        cls_score_list_af = []
        reg_dist_list_af = []
        cls_score_list_ab = []
        reg_dist_list_ab = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            reg_feat = self.reg_convs[i](reg_x)
            cls_output_ab = self.cls_preds_ab[i](cls_feat)
            reg_output_ab = self.reg_preds_ab[i](reg_feat)
            cls_output_ab = torch.sigmoid(cls_output_ab)
            cls_output_ab = cls_output_ab.reshape(b, self.na, -1, h, w
                ).permute(0, 1, 3, 4, 2)
            cls_score_list_ab.append(cls_output_ab.flatten(1, 3))
            reg_output_ab = reg_output_ab.reshape(b, self.na, -1, h, w
                ).permute(0, 1, 3, 4, 2)
            reg_output_ab[..., 2:4] = (reg_output_ab[..., 2:4].sigmoid() * 2
                ) ** 2 * self.anchors_init[i].reshape(1, self.na, 1, 1, 2).to(
                device)
            reg_dist_list_ab.append(reg_output_ab.flatten(1, 3))
            cls_output_af = self.cls_preds[i](cls_feat)
            reg_output_af = self.reg_preds[i](reg_feat)
            cls_output_af = torch.sigmoid(cls_output_af)
            cls_score_list_af.append(cls_output_af.flatten(2).permute((0, 2,
                1)))
            reg_dist_list_af.append(reg_output_af.flatten(2).permute((0, 2, 1))
                )
        cls_score_list_ab = torch.cat(cls_score_list_ab, axis=1)
        reg_dist_list_ab = torch.cat(reg_dist_list_ab, axis=1)
        cls_score_list_af = torch.cat(cls_score_list_af, axis=1)
        reg_dist_list_af = torch.cat(reg_dist_list_af, axis=1)
        return (x, cls_score_list_ab, reg_dist_list_ab, cls_score_list_af,
            reg_dist_list_af)
    else:
        device = x[0].device
        cls_score_list_af = []
        reg_dist_list_af = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            reg_feat = self.reg_convs[i](reg_x)
            cls_output_af = self.cls_preds[i](cls_feat)
            reg_output_af = self.reg_preds[i](reg_feat)
            if self.use_dfl:
                reg_output_af = reg_output_af.reshape([-1, 4, self.reg_max +
                    1, l]).permute(0, 2, 1, 3)
                reg_output_af = self.proj_conv(F.softmax(reg_output_af, dim=1))
            cls_output_af = torch.sigmoid(cls_output_af)
            if self.export:
                cls_score_list_af.append(cls_output_af)
                reg_dist_list_af.append(reg_output_af)
            else:
                cls_score_list_af.append(cls_output_af.reshape([b, self.nc, l])
                    )
                reg_dist_list_af.append(reg_output_af.reshape([b, 4, l]))
        if self.export:
            return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(
                cls_score_list_af, reg_dist_list_af))
        cls_score_list_af = torch.cat(cls_score_list_af, axis=-1).permute(0,
            2, 1)
        reg_dist_list_af = torch.cat(reg_dist_list_af, axis=-1).permute(0, 2, 1
            )
        anchor_points_af, stride_tensor_af = generate_anchors(x, self.
            stride, self.grid_cell_size, self.grid_cell_offset, device=x[0]
            .device, is_eval=True, mode='af')
        pred_bboxes_af = dist2bbox(reg_dist_list_af, anchor_points_af,
            box_format='xywh')
        pred_bboxes_af *= stride_tensor_af
        pred_bboxes = pred_bboxes_af
        cls_score_list = cls_score_list_af
        return torch.cat([pred_bboxes, torch.ones((b, pred_bboxes.shape[1],
            1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
            cls_score_list], axis=-1)
