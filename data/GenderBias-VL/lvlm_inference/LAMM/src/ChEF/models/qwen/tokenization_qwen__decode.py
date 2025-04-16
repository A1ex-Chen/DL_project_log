def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens:
    bool=False, errors: str=None, **kwargs) ->str:
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    def _decode_imgurl(img_token_ids):
        assert img_token_ids[0] == self.img_start_id and img_token_ids[-1
            ] == self.img_end_id
        img_token_ids = img_token_ids[1:-1]
        img_token_ids = img_token_ids[:img_token_ids.index(self.img_pad_id)]
        img_url = bytes(img_token_ids).decode('utf-8')
        return [self.img_start_id] + self.tokenizer.encode(img_url) + [self
            .img_end_id]
    token_ids = _replace_closed_tag(token_ids, self.img_start_id, self.
        img_end_id, _decode_imgurl)
    if skip_special_tokens:
        if kwargs.get('keep_image_special', False):
            token_ids = [i for i in token_ids if i < self.eod_id or i in
                self.image_special_tokens]
        else:
            token_ids = [i for i in token_ids if i < self.eod_id]
    return self.tokenizer.decode(token_ids, errors=errors or self.errors)
