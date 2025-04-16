def __prep_out_dist__(self, name: str) ->str:
    res: str = None
    if name == ModelDataset.OUT_DIST_UNSPLASH_HORSE:
        res = ModelDataset.get_path(ModelDataset.real_images_dir,
            'out_dist_horse')
    elif name == ModelDataset.OUT_DIST_UNSPLASH_FACE:
        res = ModelDataset.get_path(ModelDataset.real_images_dir,
            'out_dist_human_face')
    elif name == ModelDataset.OUT_DIST_FFHQ:
        res = ModelDataset.get_path(ModelDataset.real_images_dir,
            'out_dist_ffhq')
    return res
