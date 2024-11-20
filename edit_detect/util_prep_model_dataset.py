def prep_model_dataset(self, model: str, dataset: str, out_dist: str):
    res: List[str] = None
    if model == ModelDataset.MD_DDPM:
        if dataset == ModelDataset.DS_CIFAR10:
            res = [ModelDataset.MDID_GOOGLE_DDPM_CIFAR10_32
                ] + self.__prep_real_fake__(real_fodler='cifar10',
                fake_folder='cifar10_ddpm')
        elif dataset == ModelDataset.DS_CELEBA_HQ_256:
            res = [ModelDataset.MDID_GOOGLE_DDPM_CELEBA_HQ_256
                ] + self.__prep_real_fake__(real_fodler='celeba_hq_256',
                fake_folder='celeba_hq_256_ddpm')
        elif dataset == ModelDataset.DS_BEDROOM_256:
            res = [ModelDataset.MDID_GOOGLE_DDPM_BEDROOM_256
                ] + self.__prep_real_fake__(real_fodler='bedroom_256',
                fake_folder='bedroom_256_ddpm')
        elif dataset == ModelDataset.DS_CHURCH_256:
            res = [ModelDataset.MDID_GOOGLE_DDPM_CHURCH_256
                ] + self.__prep_real_fake__(real_fodler='church_256',
                fake_folder='church_256_ddpm')
        else:
            raise ValueError(
                f'Model, {model}, does not support  Dataset, {dataset}.')
    if model == ModelDataset.MD_DDPM_EMA:
        if dataset == ModelDataset.DS_CELEBA_HQ_256:
            res = [ModelDataset.MDID_GOOGLE_DDPM_EMA_CELEBA_HQ_256,
                ModelDataset.get_path(ModelDataset.real_images_dir,
                'celeba_hq_256'), ModelDataset.get_path(ModelDataset.
                fake_images_dir, 'celeba_hq_256_ddpm')]
        else:
            raise ValueError(
                f'Model, {model}, does not support Dataset, {dataset}.')
    else:
        raise ValueError(f'Model, {model}, is not supported.')
    out_dist_path: str = self.__prep_out_dist__(name=out_dist)
    return res + [out_dist_path]
