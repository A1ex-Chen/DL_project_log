def add_field(self, field, field_data):
    """Add field to box list.

    This method can be used to add related box data such as
    weights/labels, etc.

    Args:
      field: a string key to access the data via `get`
      field_data: a tensor containing the data to store in the BoxList
    """
    self.data[field] = field_data
