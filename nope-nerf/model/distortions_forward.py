def forward(self, cam_id):
    scale = self.global_scales[cam_id]
    if scale < 0.01:
        scale = torch.tensor(0.01, device=self.global_scales.device)
    if self.fix_scaleN and cam_id == self.num_cams - 1:
        scale = torch.tensor(1, device=self.global_scales.device)
    shift = self.global_shifts[cam_id]
    return scale, shift
