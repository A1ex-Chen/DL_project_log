def prep_noisy_dataloader(batch_size: int, image: Union[torch.Tensor, Image
    .Image, str, os.PathLike, pathlib.PurePath], epsilon_scale: float=1.0,
    size: int=32, num: int=1000, generator: Union[int, torch.Generator]=0,
    device: Union[str, torch.device]='cuda') ->DataLoader:
    rng: torch.Generator = set_generator(generator=generator)
    if isinstance(size, int):
        size_hw = size, size
    ds: NoisyDataset = NoisyDataset(image=image, epsilon_scale=
        epsilon_scale, size=size_hw, num=num, generator=generator)
    dl: DataLoader = DataLoader(ds, batch_size=batch_size, shuffle=True,
        generator=rng, collate_fn=lambda x: tuple(x_.to(device) for x_ in
        default_collate(x)))
    return dl
