def __init__(self, args, is_train, split, preprocess_dir, merge_classes=
    False, scale=20, full_scale=4096, resize=(480, 302), image_normalizer=
    None, noisy_rot=0.0, flip_y=0.0, rot_z=0.0, transl=False, rand_crop=
    tuple(), fliplr=0.0, color_jitter=None):
    super().__init__(split, preprocess_dir, merge_classes=merge_classes)
    self.scale = scale
    self.full_scale = full_scale
    self.noisy_rot = noisy_rot
    self.flip_y = flip_y
    self.rot_z = rot_z
    self.transl = transl
    self.resize = resize
    self.image_normalizer = image_normalizer
    if rand_crop:
        self.crop_prob = rand_crop[0]
        self.crop_dims = np.array(rand_crop[1:])
    else:
        self.crop_prob = 0.0
    self.fliplr = fliplr
    self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
    self.with_vfm = args.vfmlab
    self.is_train = is_train
