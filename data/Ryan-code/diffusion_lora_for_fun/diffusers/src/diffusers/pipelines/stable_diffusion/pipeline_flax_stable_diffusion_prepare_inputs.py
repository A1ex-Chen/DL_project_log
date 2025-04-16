def prepare_inputs(self, prompt: Union[str, List[str]]):
    if not isinstance(prompt, (str, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    text_input = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='np')
    return text_input.input_ids
