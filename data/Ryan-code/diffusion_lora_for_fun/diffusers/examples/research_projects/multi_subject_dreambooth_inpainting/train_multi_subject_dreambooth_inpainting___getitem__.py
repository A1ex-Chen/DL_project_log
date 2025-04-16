def __getitem__(self, index):
    example = {}
    img_idx = index % len(self.train_data)
    switch = random.choice([True, False])
    image = self.set_image(self.train_data[img_idx]['image'], switch)
    image_norm = self.image_normalize(image)
    tokenized_prompt = self.tokenizer(self.train_data[img_idx]['prompt'],
        padding='do_not_pad', truncation=True, max_length=self.tokenizer.
        model_max_length).input_ids
    masks = [self.set_image(self.train_data[img_idx][key], switch) for key in
        self.train_data[img_idx] if 'mask' in key]
    example['PIL_image'] = image
    example['instance_image'] = image_norm
    example['instance_prompt_id'] = tokenized_prompt
    example['instance_masks'] = masks
    return example
