def __init__(self, train_data_root, tokenizer, size=512):
    self.size = size
    self.tokenizer = tokenizer
    self.ref_data_root = Path(train_data_root) / 'ref'
    self.target_image = Path(train_data_root) / 'target' / 'target.png'
    self.target_mask = Path(train_data_root) / 'target' / 'mask.png'
    if not (self.ref_data_root.exists() and self.target_image.exists() and
        self.target_mask.exists()):
        raise ValueError("Train images root doesn't exists.")
    self.train_images_path = list(self.ref_data_root.iterdir()) + [self.
        target_image]
    self.num_train_images = len(self.train_images_path)
    self.train_prompt = 'a photo of sks'
    self.transform = transforms_v2.Compose([transforms_v2.ToImage(),
        transforms_v2.RandomResize(size, int(1.125 * size)), transforms_v2.
        RandomCrop(size), transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize([0.5], [0.5])])
