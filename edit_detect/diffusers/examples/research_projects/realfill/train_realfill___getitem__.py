def __getitem__(self, index):
    example = {}
    image = Image.open(self.train_images_path[index])
    image = exif_transpose(image)
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    if index < len(self) - 1:
        weighting = Image.new('L', image.size)
    else:
        weighting = Image.open(self.target_mask)
        weighting = exif_transpose(weighting)
    image, weighting = self.transform(image, weighting)
    example['images'], example['weightings'] = image, weighting < 0
    if random.random() < 0.1:
        example['masks'] = torch.ones_like(example['images'][0:1, :, :])
    else:
        example['masks'] = make_mask(example['images'], self.size)
    example['conditioning_images'] = example['images'] * (example['masks'] <
        0.5)
    train_prompt = '' if random.random() < 0.1 else self.train_prompt
    example['prompt_ids'] = self.tokenizer(train_prompt, truncation=True,
        padding='max_length', max_length=self.tokenizer.model_max_length,
        return_tensors='pt').input_ids
    return example
