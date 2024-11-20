def __init__(self, concepts_list, tokenizer, size=512, mask_size=64,
    center_crop=False, with_prior_preservation=False, num_class_images=200,
    hflip=False, aug=True):
    self.size = size
    self.mask_size = mask_size
    self.center_crop = center_crop
    self.tokenizer = tokenizer
    self.interpolation = Image.BILINEAR
    self.aug = aug
    self.instance_images_path = []
    self.class_images_path = []
    self.with_prior_preservation = with_prior_preservation
    for concept in concepts_list:
        inst_img_path = [(x, concept['instance_prompt']) for x in Path(
            concept['instance_data_dir']).iterdir() if x.is_file()]
        self.instance_images_path.extend(inst_img_path)
        if with_prior_preservation:
            class_data_root = Path(concept['class_data_dir'])
            if os.path.isdir(class_data_root):
                class_images_path = list(class_data_root.iterdir())
                class_prompt = [concept['class_prompt'] for _ in range(len(
                    class_images_path))]
            else:
                with open(class_data_root, 'r') as f:
                    class_images_path = f.read().splitlines()
                with open(concept['class_prompt'], 'r') as f:
                    class_prompt = f.read().splitlines()
            class_img_path = list(zip(class_images_path, class_prompt))
            self.class_images_path.extend(class_img_path[:num_class_images])
    random.shuffle(self.instance_images_path)
    self.num_instance_images = len(self.instance_images_path)
    self.num_class_images = len(self.class_images_path)
    self._length = max(self.num_class_images, self.num_instance_images)
    self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)
    self.image_transforms = transforms.Compose([self.flip, transforms.
        Resize(size, interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.CenterCrop(size) if center_crop else transforms.
        RandomCrop(size), transforms.ToTensor(), transforms.Normalize([0.5],
        [0.5])])
