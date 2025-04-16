def as_tensor_dict(self, fields=None):
    """Retrieves specified fields as a dictionary of tensors.

    Args:
      fields: (optional) list of fields to return in the dictionary.
        If None (default), all fields are returned.

    Returns:
      tensor_dict: A dictionary of tensors specified by fields.

    Raises:
      ValueError: if specified field is not contained in boxlist.
    """
    tensor_dict = {}
    if fields is None:
        fields = self.get_all_fields()
    for field in fields:
        if not self.has_field(field):
            raise ValueError('boxlist must contain all specified fields')
        tensor_dict[field] = self.get_field(field)
    return tensor_dict
