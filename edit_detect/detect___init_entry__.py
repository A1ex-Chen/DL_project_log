def __init_entry__(self, image: Union[torch.Tensor, Image.Image, str, os.
    PathLike, pathlib.PurePath], num: int, epsilon_scale: float):
    proc_img = self.__process_img__(image=image)
    noise = torch.randn(size=(num, *proc_img.shape), generator=self.
        __generator__) * epsilon_scale
    return {NoisyDataset.IMAGE_KEY: proc_img, NoisyDataset.NOISE_KEY: noise,
        NoisyDataset.LATENT_KEY: {}, NoisyDataset.RECONST_KEY: {},
        NoisyDataset.RESIDUAL_KEY: {}}
