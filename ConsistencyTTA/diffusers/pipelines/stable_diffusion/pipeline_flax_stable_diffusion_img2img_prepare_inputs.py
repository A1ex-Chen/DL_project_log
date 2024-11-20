def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.
    Image, List[Image.Image]]):
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
    processed_images = jnp.concatenate([preprocess(img, jnp.float32) for
        img in image])
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='np')
    return text_input.input_ids, processed_images
