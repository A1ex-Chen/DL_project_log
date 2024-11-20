def to_list_format(self, text: str):
    text = unicodedata.normalize('NFC', text)
    token_ids = self.tokenizer.encode(text, allowed_special=set(self.
        IMAGE_ST + (ENDOFTEXT,)))

    def _encode_vl_info(tokens):
        if len(tokens) == 0:
            return []
        if tokens[0] == self.img_start_id and tokens[-1] == self.img_end_id:
            key = 'image'
        elif tokens[0] == self.ref_start_id and tokens[-1] == self.ref_end_id:
            key = 'ref'
        elif tokens[0] == self.box_start_id and tokens[-1] == self.box_end_id:
            key = 'box'
        elif tokens[0] == self.quad_start_id and tokens[-1
            ] == self.quad_end_id:
            key = 'quad'
        else:
            _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
            return [{'text': b''.join(map(_tobytes, map(self.decoder.get,
                tokens))).decode('utf-8')}]
        _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
        val = b''.join(map(_tobytes, map(self.decoder.get, tokens[1:-1]))
            ).decode('utf-8')
        return [{key: val}]
    return _replace_closed_tag(token_ids, (self.img_start_id, self.
        ref_start_id, self.box_start_id, self.quad_start_id), (self.
        img_end_id, self.ref_end_id, self.box_end_id, self.quad_end_id),
        _encode_vl_info, _encode_vl_info)
