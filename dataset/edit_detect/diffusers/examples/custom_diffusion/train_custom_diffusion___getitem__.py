def __getitem__(self, index):
    example = {}
    instance_image, instance_prompt = self.instance_images_path[index %
        self.num_instance_images]
    instance_image = Image.open(instance_image)
    if not instance_image.mode == 'RGB':
        instance_image = instance_image.convert('RGB')
    instance_image = self.flip(instance_image)
    random_scale = self.size
    if self.aug:
        random_scale = np.random.randint(self.size // 3, self.size + 1
            ) if np.random.uniform() < 0.66 else np.random.randint(int(1.2 *
            self.size), int(1.4 * self.size))
    instance_image, mask = self.preprocess(instance_image, random_scale,
        self.interpolation)
    if random_scale < 0.6 * self.size:
        instance_prompt = np.random.choice(['a far away ', 'very small ']
            ) + instance_prompt
    elif random_scale > self.size:
        instance_prompt = np.random.choice(['zoomed in ', 'close up ']
            ) + instance_prompt
    example['instance_images'] = torch.from_numpy(instance_image).permute(2,
        0, 1)
    example['mask'] = torch.from_numpy(mask)
    example['instance_prompt_ids'] = self.tokenizer(instance_prompt,
        truncation=True, padding='max_length', max_length=self.tokenizer.
        model_max_length, return_tensors='pt').input_ids
    if self.with_prior_preservation:
        class_image, class_prompt = self.class_images_path[index % self.
            num_class_images]
        class_image = Image.open(class_image)
        if not class_image.mode == 'RGB':
            class_image = class_image.convert('RGB')
        example['class_images'] = self.image_transforms(class_image)
        example['class_mask'] = torch.ones_like(example['mask'])
        example['class_prompt_ids'] = self.tokenizer(class_prompt,
            truncation=True, padding='max_length', max_length=self.
            tokenizer.model_max_length, return_tensors='pt').input_ids
    return example
