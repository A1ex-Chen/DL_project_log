def set_field(self, field, value):
    """Sets the value of a field.

    Updates the field of a box_list with a given value.

    Args:
      field: (string) name of the field to set value.
      value: the value to assign to the field.

    Raises:
      ValueError: if the box_list does not have specified field.
    """
    if not self.has_field(field):
        raise ValueError('field %s does not exist' % field)
    self.data[field] = value
