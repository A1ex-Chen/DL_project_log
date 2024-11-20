def center_mse_series(pred_orig_images: List[torch.Tensor]):
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    pred_orig_images_mean: torch.Tensor = pred_orig_images.mean(dim=1,
        keepdim=True)
    pred_orig_images_mean_stck = pred_orig_images_mean.repeat(1, n, *([1] *
        (len(pred_orig_images.shape) - 2)))
    residual: torch.Tensor = ((pred_orig_images_mean_stck -
        pred_orig_images) ** 2).sqrt().mean(list(range(
        pred_orig_images_mean_stck.dim()))[2:]).transpose(0, 1)
    return residual
