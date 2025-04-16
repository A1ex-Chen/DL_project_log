def prep_real_fake_dataloader(batch_size: int, root: Union[str, os.PathLike,
    pathlib.PurePath], size: Union[int, Tuple[int, int], List[int]],
    dir_label_map: Dict[str, int]={'real': 0, 'fake': 1}, generator: Union[
    int, torch.Generator]=0, image_exts: List[str]=['png, jpg, jpeg, webp'],
    ext_case_sensitive: bool=False) ->DataLoader:
    rng: torch.Generator = set_generator(generator=generator)
    ds: RealFakeDataset = RealFakeDataset(root=root, size=size,
        dir_label_map=dir_label_map, image_exts=image_exts, generator=
        generator, ext_case_sensitive=ext_case_sensitive)
    dl: DataLoader = DataLoader(ds, batch_size=batch_size, shuffle=True,
        generator=rng)
    return dl
