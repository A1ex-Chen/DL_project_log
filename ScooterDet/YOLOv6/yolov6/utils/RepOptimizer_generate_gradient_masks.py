def generate_gradient_masks(self, scales_by_idx, conv3x3_by_idx, cpu_mode=False
    ):
    self.grad_mask_map = {}
    for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
        para = conv3x3.weight
        if len(scales) == 2:
            mask = torch.ones_like(para, device=scales[0].device) * (scales
                [1] ** 2).view(-1, 1, 1, 1)
            mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1],
                1, 1, device=scales[0].device) * (scales[0] ** 2).view(-1, 
                1, 1, 1)
        else:
            mask = torch.ones_like(para, device=scales[0].device) * (scales
                [2] ** 2).view(-1, 1, 1, 1)
            mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1],
                1, 1, device=scales[0].device) * (scales[1] ** 2).view(-1, 
                1, 1, 1)
            ids = np.arange(para.shape[1])
            assert para.shape[1] == para.shape[0]
            mask[ids, ids, 1:2, 1:2] += 1.0
        if cpu_mode:
            self.grad_mask_map[para] = mask
        else:
            self.grad_mask_map[para] = mask.cuda()
