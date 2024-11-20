def __getitem__(self, i):
    example = {}
    image = Image.open(self.image_paths[i % self.num_images])
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    placeholder_string = self.placeholder_token
    text = random.choice(self.templates).format(placeholder_string)
    example['original_size'] = image.height, image.width
    image = image.resize((self.size, self.size), resample=self.interpolation)
    if self.center_crop:
        y1 = max(0, int(round((image.height - self.size) / 2.0)))
        x1 = max(0, int(round((image.width - self.size) / 2.0)))
        image = self.crop(image)
    else:
        y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
        image = transforms.functional.crop(image, y1, x1, h, w)
    example['crop_top_left'] = y1, x1
    example['input_ids_1'] = self.tokenizer_1(text, padding='max_length',
        truncation=True, max_length=self.tokenizer_1.model_max_length,
        return_tensors='pt').input_ids[0]
    example['input_ids_2'] = self.tokenizer_2(text, padding='max_length',
        truncation=True, max_length=self.tokenizer_2.model_max_length,
        return_tensors='pt').input_ids[0]
    img = np.array(image).astype(np.uint8)
    image = Image.fromarray(img)
    image = self.flip_transform(image)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    example['pixel_values'] = torch.from_numpy(image).permute(2, 0, 1)
    return example
