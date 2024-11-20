def reinitialize(self, scales_by_idx, conv3x3_by_idx, use_identity_scales):
    for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
        in_channels = conv3x3.in_channels
        out_channels = conv3x3.out_channels
        kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1, device=conv3x3
            .weight.device)
        if len(scales) == 2:
            conv3x3.weight.data = conv3x3.weight * scales[1].view(-1, 1, 1, 1
                ) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[0].view(
                -1, 1, 1, 1)
        else:
            assert len(scales) == 3
            assert in_channels == out_channels
            identity = torch.from_numpy(np.eye(out_channels, dtype=np.
                float32).reshape(out_channels, out_channels, 1, 1)).to(conv3x3
                .weight.device)
            conv3x3.weight.data = conv3x3.weight * scales[2].view(-1, 1, 1, 1
                ) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[1].view(
                -1, 1, 1, 1)
            if use_identity_scales:
                identity_scale_weight = scales[0]
                conv3x3.weight.data += F.pad(identity *
                    identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
            else:
                conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])
