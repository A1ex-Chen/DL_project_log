def get_extra_fields(self):
    """Returns all non-box fields (i.e., everything not named 'boxes')."""
    return [k for k in self.data.keys() if k != 'boxes']
