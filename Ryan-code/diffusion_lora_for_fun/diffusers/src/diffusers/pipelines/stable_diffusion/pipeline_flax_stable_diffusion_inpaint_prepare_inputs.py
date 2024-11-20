def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.
    Image, List[Image.Image]], mask: Union[Image.Image, List[Image.Image]]):
    if not isinstance(prompt, (str, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if not isinstance(image, (Image.Image, list)):
        raise ValueError(
            f'image has to be of type `PIL.Image.Image` or list but is {type(image)}'
            )
    if isinstance(image, Image.Image):
        image = [image]
    if not isinstance(mask, (Image.Image, list)):
        raise ValueError(
            f'image has to be of type `PIL.Image.Image` or list but is {type(image)}'
            )
    if isinstance(mask, Image.Image):
        mask = [mask]
    processed_images = jnp.concatenate([preprocess_image(img, jnp.float32) for
        img in image])
    processed_masks = jnp.concatenate([preprocess_mask(m, jnp.float32) for
        m in mask])
    processed_masks = processed_masks.at[processed_masks < 0.5].set(0)
    processed_masks = processed_masks.at[processed_masks >= 0.5].set(1)
    processed_masked_images = processed_images * (processed_masks < 0.5)
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='np')
    return text_input.input_ids, processed_masked_images, processed_masks
