def __prep_real_fake__(self, real_fodler: str, fake_folder: str):
    return [ModelDataset.get_path(ModelDataset.real_images_dir, real_fodler
        ), ModelDataset.get_path(ModelDataset.fake_images_dir, fake_folder)]
