def __init__(self, args, split, preprocess_dir, nuscenes_dir='',
    pselab_paths=None, merge_classes=False, scale=20, full_scale=4096,
    resize=(400, 225), image_normalizer=None, noisy_rot=0.0, flip_x=0.0,
    rot_z=0.0, transl=False, fliplr=0.0, color_jitter=None, output_orig=False):
    super().__init__(split, preprocess_dir, merge_classes=merge_classes,
        pselab_paths=pselab_paths)
    self.nuscenes_dir = nuscenes_dir
    self.output_orig = output_orig
    self.scale = scale
    self.full_scale = full_scale
    self.noisy_rot = noisy_rot
    self.flip_x = flip_x
    self.rot_z = rot_z
    self.transl = transl
    self.resize = resize
    self.image_normalizer = image_normalizer
    self.fliplr = fliplr
    self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
    self.with_vfm = args.vfmlab
