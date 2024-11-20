def __init__(self, data_root, tokenizer, learnable_property='object', size=
    512, repeats=100, interpolation='bicubic', flip_p=0.5, set='train',
    placeholder_token='*', center_crop=False):
    self.data_root = data_root
    self.tokenizer = tokenizer
    self.learnable_property = learnable_property
    self.size = size
    self.placeholder_token = placeholder_token
    self.center_crop = center_crop
    self.flip_p = flip_p
    self.image_paths = [os.path.join(self.data_root, file_path) for
        file_path in os.listdir(self.data_root)]
    self.num_images = len(self.image_paths)
    self._length = self.num_images
    if set == 'train':
        self._length = self.num_images * repeats
    self.interpolation = {'linear': PIL_INTERPOLATION['linear'], 'bilinear':
        PIL_INTERPOLATION['bilinear'], 'bicubic': PIL_INTERPOLATION[
        'bicubic'], 'lanczos': PIL_INTERPOLATION['lanczos']}[interpolation]
    self.templates = (imagenet_style_templates_small if learnable_property ==
        'style' else imagenet_templates_small)
    self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
