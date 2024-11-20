def __getitem__(self, i):
    example = {}
    image = Image.open(self.image_paths[i % self.num_images])
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    placeholder_string = self.placeholder_token
    text = random.choice(self.templates).format(placeholder_string)
    example['input_ids'] = self.tokenizer.encode(text, padding='max_length',
        truncation=True, max_length=self.tokenizer.model_max_length,
        return_tensors='pt', vector_shuffle=self.vector_shuffle,
        prop_tokens_to_load=self.prop_tokens_to_load if self.
        progressive_tokens else 1.0)[0]
    img = np.array(image).astype(np.uint8)
    if self.center_crop:
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w +
            crop) // 2]
    image = Image.fromarray(img)
    image = image.resize((self.size, self.size), resample=self.interpolation)
    image = self.flip_transform(image)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    example['pixel_values'] = torch.from_numpy(image).permute(2, 0, 1)
    return example
