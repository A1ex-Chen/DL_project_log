@torch.no_grad()
def compute_flow_timestep(pipeline: DiffusionPipeline, timestep: Union[int,
    torch.IntTensor, torch.LongTensor], batch_size: int, image: Union[torch
    .Tensor, Image.Image, str, os.PathLike, pathlib.PurePath], num: int=
    1000, generator: Union[int, torch.Generator]=0, device: Union[str,
    torch.device]='cuda'):
    pipeline = pipeline.to(device)
    dl = prep_noisy_dataloader(batch_size=batch_size, image=image, size=
        pipeline.unet.config.sample_size, num=num, generator=generator,
        device=device)
    if isinstance(timestep, int):
        timestep: torch.LongTensor = torch.LongTensor([timestep] * batch_size
            ).to(device)
    timestep: torch.LongTensor = timestep.long().to(device)
    timestep_dist: torch.Tensor = torch.tensor([1.0] * len(timestep)).to(device
        )
    alpha_prod_t = pipeline.scheduler.alphas_cumprod.to(device)[timestep.
        unique()]
    beta_prod_t = 1 - alpha_prod_t
    print(
        f'Noise Scale(Beta Bar T): {beta_prod_t ** 0.5}, Content Scale(Alpha Bar T): {alpha_prod_t ** 0.5}'
        )
    eps_dis_ls: List[torch.Tensor] = []
    for img, noise in dl:
        t: torch.IntTensor = timestep[torch.multinomial(timestep_dist, len(
            img), replacement=True)]
        pred_epsilon = reconstruct_epsilon(pipeline=pipeline, x0=img, t=t,
            noise=noise, generator=generator)
        eps_dis = epsilon_distance(epsilon=noise, pred_epsilon=pred_epsilon)
        eps_dis_ls.append(eps_dis)
    eps_dis_ls: torch.Tensor = torch.cat(eps_dis_ls)
    return eps_dis_ls, eps_dis_ls.mean(), eps_dis_ls.var()
