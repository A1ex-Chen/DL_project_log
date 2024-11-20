@torch.no_grad()
def compute_gaussian_reconstruct(pipeline: DiffusionPipeline, noise_scale:
    Union[float, torch.FloatTensor, torch.Tensor], batch_size: int, image:
    Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath],
    num: int=1000, generator: Union[int, torch.Generator]=0, device: Union[
    str, torch.device]='cuda'):
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.
        scheduler.config)
    pipeline = pipeline.to(device)
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=
        pipeline.unet.config.sample_size, num=num, generator=generator,
        device=device)
    if isinstance(noise_scale, float):
        noise_scale: torch.FloatTensor = torch.FloatTensor([noise_scale] *
            batch_size).to(device)
    noise_scales_dist: torch.Tensor = torch.tensor([1.0] * len(noise_scale)
        ).to(device)
    x0_dis_ls: List[torch.Tensor] = []
    for img, noise in dl:
        noise_scale_sample: torch.FloatTensor = noise_scale[torch.
            multinomial(noise_scales_dist, len(img), replacement=True)]
        pred_x0 = reconstruct_x0_n_steps(pipeline=pipeline, x0=img,
            noise_scale=noise_scale_sample, noise=noise, generator=generator
            ).to(device)
        """
        MSE distance metric
        """
        """
        High-Pass kernel and L2 Metric
        """
        fft_fn = DistanceFn.embed_fft_1d_fn(thres=0.3, nd=-3)
        img_fft, pred_x0_fft = fft_fn(img), fft_fn(pred_x0)
        norm_dis = DistanceFn.norm_fn(loss_type=DistanceFn.NORM_L2)(img_fft,
            pred_x0_fft)
        x0_dis = DistanceFn.reduce_fn(reduce_type=DistanceFn.REDUCE_MEAN, nd=-1
            )(norm_dis)
        x0_dis_ls.append(x0_dis)
    x0_dis_ls: torch.Tensor = torch.cat(x0_dis_ls)
    return x0_dis_ls, x0_dis_ls.mean(), x0_dis_ls.var()
