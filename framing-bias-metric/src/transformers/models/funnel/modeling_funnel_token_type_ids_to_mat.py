def token_type_ids_to_mat(self, token_type_ids):
    """Convert `token_type_ids` to `token_type_mat`."""
    token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
    cls_ids = token_type_ids == self.cls_token_type_id
    cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
    return cls_mat | token_type_mat
