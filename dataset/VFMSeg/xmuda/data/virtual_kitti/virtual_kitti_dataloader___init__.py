def __init__(self, args, is_train, split, preprocess_dir, virtual_kitti_dir
    ='', merge_classes=False, scale=20, full_scale=4096, image_normalizer=
    None, noisy_rot=0.0, flip_y=0.0, rot_z=0.0, transl=False, downsample=(-
    1,), crop_size=tuple(), bottom_crop=False, rand_crop=tuple(), fliplr=
    0.0, color_jitter=None, random_weather=tuple()):
    super().__init__(split, preprocess_dir, merge_classes=merge_classes)
    self.virtual_kitti_dir = virtual_kitti_dir
    self.scale = scale
    self.full_scale = full_scale
    self.noisy_rot = noisy_rot
    self.flip_y = flip_y
    self.rot_z = rot_z
    self.transl = transl
    assert isinstance(downsample, tuple)
    if len(downsample) == 1:
        self.downsample = downsample[0]
    elif len(downsample) == 2:
        self.downsample = downsample
    else:
        NotImplementedError(
            'Downsample must be either a tuple of (num_points,) or (min_points, max_points),such that a different number of points can be sampled randomly for each example.'
            )
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
    self.random_weather = random_weather
    self.fliplr = fliplr
    self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
    self.with_vfm = args.vfmlab
    self.is_train = is_train
