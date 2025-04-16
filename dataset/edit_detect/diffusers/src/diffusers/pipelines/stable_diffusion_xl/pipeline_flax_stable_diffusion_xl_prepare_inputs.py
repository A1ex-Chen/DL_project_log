def prepare_inputs(self, prompt: Union[str, List[str]]):
    if not isinstance(prompt, (str, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    inputs = []
    for tokenizer in [self.tokenizer, self.tokenizer_2]:
        text_inputs = tokenizer(prompt, padding='max_length', max_length=
            self.tokenizer.model_max_length, truncation=True,
            return_tensors='np')
        inputs.append(text_inputs.input_ids)
    inputs = jnp.stack(inputs, axis=1)
    return inputs
