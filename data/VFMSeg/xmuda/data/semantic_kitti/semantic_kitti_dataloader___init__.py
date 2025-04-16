def __init__(self, args, is_train, split, preprocess_dir,
    semantic_kitti_dir='', pselab_paths=None, merge_classes_style=None,
    scale=20, full_scale=4096, image_normalizer=None, noisy_rot=0.0, flip_y
    =0.0, rot_z=0.0, transl=False, crop_size=tuple(), bottom_crop=False,
    rand_crop=tuple(), fliplr=0.0, color_jitter=None, output_orig=False):
    super().__init__(split, preprocess_dir, merge_classes_style=
        merge_classes_style, pselab_paths=pselab_paths)
    self.semantic_kitti_dir = semantic_kitti_dir
    self.output_orig = output_orig
    self.scale = scale
    self.full_scale = full_scale
    self.noisy_rot = noisy_rot
    self.flip_y = flip_y
    self.rot_z = rot_z
    self.transl = transl
    self.image_normalizer = image_normalizer
    self.crop_size = crop_size
    if self.crop_size:
        assert bottom_crop != bool(rand_crop
            ), 'Exactly one crop method needs to be active if crop size is provided!'
    else:
        assert not bottom_crop and not rand_crop, 'No crop size, but crop method is provided is provided!'
    self.bottom_crop = bottom_crop
    self.rand_crop = np.array(rand_crop)
    assert len(self.rand_crop) in [0, 4]
    self.fliplr = fliplr
    self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
    self.with_vfm = args.vfmlab
    self.is_train = is_train
