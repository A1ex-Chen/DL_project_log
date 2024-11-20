@torch.no_grad()
def compute_direct_reconst(pipeline: DiffusionPipeline, timestep: int,
    batch_size: int, image: Union[torch.Tensor, Image.Image, str, os.
    PathLike, pathlib.PurePath], num: int=1000, generator: Union[int, torch
    .Generator]=0, device: Union[str, torch.device]='cuda', recorder:
    SafetensorRecorder=None, label: str=None):
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=
        pipeline.unet.config.sample_size, num=num, generator=generator,
        device=device)
    x0_dis_ls: List[torch.Tensor] = []
    x0_dis_trend_ls: List[torch.Tensor] = []
    for img, noise in dl:
        pred_x0, pred_orig_images = reconstruct_x0_direct_n_steps(pipeline=
            pipeline, x0=img, timestep=timestep, noise=noise, generator=
            generator)
        img, noise, pred_x0, pred_orig_images = img.detach().cpu(
            ), noise.detach().cpu(), pred_x0.detach().cpu(
            ), pred_orig_images.detach().cpu()
        x0_dis = mse(x0=img, pred_x0=pred_x0)
        x0_dis_trend = mse_series(x0=img.detach().cpu(), pred_orig_images=
            pred_orig_images.detach().cpu())
        pred_dis_traj = mse_traj(pred_orig_images=pred_orig_images.detach()
            .cpu())
        x0_dis_ls.append(x0_dis)
        x0_dis_trend_ls.append(x0_dis_trend)
        pred_orig_images_trans: torch.Tensor = pred_orig_images.transpose(0, 1)
        noisy_image: torch.Tensor = get_xt_by_t(pipeline=pipeline, x0=img,
            t=timestep, noise=noise)
        print(
            f'x0_dis_trend: {x0_dis_trend.shape}, pred_dis_traj: {pred_dis_traj.shape}'
            )
        print(
            f'img: {img.shape}, x0_dis: {x0_dis.shape}, noise: {noise.shape}, noisy_image: {noisy_image.shape}, pred_orig_images: {pred_orig_images_trans.shape}, pred_x0: {pred_x0.shape}, timestep: {timestep}, label: {label}'
            )
        recorder.batch_update(images=img.cpu(), noisy_images=noisy_image.
            cpu(), reconsts=pred_x0.cpu(), residuals=x0_dis_trend.cpu(),
            traj_residuals=pred_dis_traj.cpu(), noises=noise.cpu(),
            timestep=timestep, label=label)
    x0_dis_ls: torch.Tensor = torch.cat(x0_dis_ls)
    x0_dis_trend_ls: torch.Tensor = torch.cat(x0_dis_trend_ls)
    print(f'x0_dis_trend_ls: {x0_dis_trend_ls.shape}')
    return x0_dis_ls, x0_dis_ls.mean(), x0_dis_ls.var(
        ), x0_dis_trend_ls, recorder
