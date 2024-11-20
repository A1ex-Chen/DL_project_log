def get_field(self, field):
    """Accesses a box collection and associated fields.

    This function returns specified field with object; if no field is specified,
    it returns the box coordinates.

    Args:
      field: this optional string parameter can be used to specify
        a related field to be accessed.

    Returns:
      a tensor representing the box collection or an associated field.

    Raises:
      ValueError: if invalid field
    """
    if not self.has_field(field):
        raise ValueError('field ' + str(field) + ' does not exist')
    return self.data[field]
