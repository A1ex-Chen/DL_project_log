def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
    """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
    for field in boxlist_to_copy_from.get_extra_fields():
        boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(
            field))
    return boxlist_to_copy_to
