def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False,
    prop_tokens_to_load=1.0):
    """
        Here, we replace the placeholder tokens in text recorded in token_map so that the text_encoder
        can encode them
        vector_shuffle was inspired by https://github.com/rinongal/textual_inversion/pull/119
        where shuffling tokens were found to force the model to learn the concepts more descriptively.
        """
    if isinstance(text, list):
        output = []
        for i in range(len(text)):
            output.append(self.replace_placeholder_tokens_in_text(text[i],
                vector_shuffle=vector_shuffle))
        return output
    for placeholder_token in self.token_map:
        if placeholder_token in text:
            tokens = self.token_map[placeholder_token]
            tokens = tokens[:1 + int(len(tokens) * prop_tokens_to_load)]
            if vector_shuffle:
                tokens = copy.copy(tokens)
                random.shuffle(tokens)
            text = text.replace(placeholder_token, ' '.join(tokens))
    return text
