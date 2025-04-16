def mse_series(x0: torch.Tensor, pred_orig_images: List[torch.Tensor]):
    t: int = len(pred_orig_images)
    n: int = pred_orig_images.shape[1]
    print(f'x0: {x0.shape}, pred_orig_images: {pred_orig_images.shape}')
    x0_stck = x0.repeat(t, *([1] * (len(pred_orig_images.shape) - 1)))
    print(
        f't: {t}, n: {n}, x0_stck: {x0_stck.shape}, pred_orig_images: {pred_orig_images.shape}'
        )
    residual: torch.Tensor = ((x0_stck - pred_orig_images) ** 2).sqrt().mean(
        list(range(x0_stck.dim()))[2:]).transpose(0, 1)
    return residual
