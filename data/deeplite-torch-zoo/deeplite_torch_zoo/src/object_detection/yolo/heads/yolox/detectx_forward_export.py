def forward_export(self, x):
    cls_xs = x[0::2]
    reg_xs = x[1::2]
    outputs = []
    for k, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride,
        cls_xs, reg_xs)):
        cls_output = self.cls_preds[k](cls_x)
        reg_output = self.reg_preds[k](reg_x)
        obj_output = self.obj_preds[k](reg_x)
        output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.
            sigmoid()], 1)
        outputs.append(output)
    outputs = torch.cat([out.flatten(start_dim=2) for out in outputs], dim=2
        ).permute(0, 2, 1)
    return outputs
