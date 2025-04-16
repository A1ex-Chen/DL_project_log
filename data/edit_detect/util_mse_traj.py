def mse_traj(pred_orig_images: List[torch.Tensor]):
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    residual: torch.Tensor = ((pred_orig_images - pred_orig_images.roll(
        shifts=1))[:-1] ** 2).sqrt().mean(list(range(pred_orig_images.dim()
        ))[2:]).transpose(0, 1)
    return residual
