def _encode_imgurl(img_tokens):
    assert img_tokens[0] == self.image_start_tag and img_tokens[-1
        ] == self.image_end_tag
    img_tokens = img_tokens[1:-1]
    img_url = b''.join(img_tokens)
    out_img_tokens = list(map(self.decoder.get, img_url))
    if len(out_img_tokens) > IMG_TOKEN_SPAN:
        raise ValueError('The content in {}..{} is too long'.format(self.
            image_start_tag, self.image_end_tag))
    out_img_tokens.extend([self.image_pad_tag] * (IMG_TOKEN_SPAN - len(
        out_img_tokens)))
    out_img_tokens = [self.image_start_tag] + out_img_tokens + [self.
        image_end_tag]
    return out_img_tokens
