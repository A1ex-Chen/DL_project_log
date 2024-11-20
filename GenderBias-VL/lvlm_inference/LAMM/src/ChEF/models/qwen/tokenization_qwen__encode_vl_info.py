def _encode_vl_info(tokens):
    if len(tokens) == 0:
        return []
    if tokens[0] == self.img_start_id and tokens[-1] == self.img_end_id:
        key = 'image'
    elif tokens[0] == self.ref_start_id and tokens[-1] == self.ref_end_id:
        key = 'ref'
    elif tokens[0] == self.box_start_id and tokens[-1] == self.box_end_id:
        key = 'box'
    elif tokens[0] == self.quad_start_id and tokens[-1] == self.quad_end_id:
        key = 'quad'
    else:
        _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
        return [{'text': b''.join(map(_tobytes, map(self.decoder.get,
            tokens))).decode('utf-8')}]
    _tobytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
    val = b''.join(map(_tobytes, map(self.decoder.get, tokens[1:-1]))).decode(
        'utf-8')
    return [{key: val}]
