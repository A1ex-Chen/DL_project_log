def tokenize(self, text: str, allowed_special: Union[Set, str]='all',
    disallowed_special: Union[Collection, str]=(), **kwargs) ->List[Union[
    bytes, str]]:
    """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
    tokens = []
    text = unicodedata.normalize('NFC', text)
    for t in self.tokenizer.encode(text, allowed_special=allowed_special,
        disallowed_special=disallowed_special):
        tokens.append(self.decoder[t])

    def _encode_imgurl(img_tokens):
        assert img_tokens[0] == self.image_start_tag and img_tokens[-1
            ] == self.image_end_tag
        img_tokens = img_tokens[1:-1]
        img_url = b''.join(img_tokens)
        out_img_tokens = list(map(self.decoder.get, img_url))
        if len(out_img_tokens) > IMG_TOKEN_SPAN:
            raise ValueError('The content in {}..{} is too long'.format(
                self.image_start_tag, self.image_end_tag))
        out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(
            out_img_tokens)))
        out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.
            image_end_tag]
        return out_img_tokens
    return _replace_closed_tag(tokens, self.image_start_tag, self.
        image_end_tag, _encode_imgurl)
