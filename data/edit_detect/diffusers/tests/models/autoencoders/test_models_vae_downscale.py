def downscale(x):
    return torch.nn.functional.avg_pool2d(x, model.spatial_scale_factor)
