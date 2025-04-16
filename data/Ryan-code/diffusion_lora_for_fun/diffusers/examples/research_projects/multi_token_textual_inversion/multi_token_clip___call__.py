def __call__(self, text, *args, vector_shuffle=False, prop_tokens_to_load=
    1.0, **kwargs):
    return super().__call__(self.replace_placeholder_tokens_in_text(text,
        vector_shuffle=vector_shuffle, prop_tokens_to_load=
        prop_tokens_to_load), *args, **kwargs)
