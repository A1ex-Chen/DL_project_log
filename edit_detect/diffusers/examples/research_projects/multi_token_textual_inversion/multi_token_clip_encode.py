def encode(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0,
    **kwargs):
    return super().encode(self.replace_placeholder_tokens_in_text(text,
        vector_shuffle=vector_shuffle, prop_tokens_to_load=
        prop_tokens_to_load), *args, **kwargs)
