def append_message(self, role: str, message: str, *, boxes=None, points=
    None, boxes_seq=None, points_seq=None):
    """Append a new message."""
    assert role in self.roles

    def convert_idx(objs_seq, objs_value, get_obj_idx_func):
        if objs_seq is None:
            return None
        ret = []
        for objs_idx in objs_seq:
            new_objs_idx = []
            for idx in objs_idx:
                new_idx = get_obj_idx_func(objs_value[idx])
                new_objs_idx.append(new_idx)
            ret.append(tuple(new_objs_idx))
        return tuple(ret)
    boxes_seq = convert_idx(boxes_seq, boxes, self._get_box_idx)
    points_seq = convert_idx(points_seq, points, self._get_point_idx)
    if self.image is not None:
        previous_message_has_image_placeholder = any('<image>' in item[
            'value'] for item in self.conversations)
        if (not previous_message_has_image_placeholder and '<image>' not in
            message):
            message = '<image> ' + message
        if previous_message_has_image_placeholder and '<image>' in message:
            message = message.replace('<image>', '')
    self.conversations.append({'from': role, 'value': message, 'boxes_seq':
        copy.deepcopy(boxes_seq), 'points_seq': copy.deepcopy(points_seq)})
