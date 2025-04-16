def _decode_imgurl(img_token_ids):
    assert img_token_ids[0] == self.img_start_id and img_token_ids[-1
        ] == self.img_end_id
    img_token_ids = img_token_ids[1:-1]
    img_token_ids = img_token_ids[:img_token_ids.index(self.img_pad_id)]
    img_url = bytes(img_token_ids).decode('utf-8')
    return [self.img_start_id] + self.tokenizer.encode(img_url) + [self.
        img_end_id]
