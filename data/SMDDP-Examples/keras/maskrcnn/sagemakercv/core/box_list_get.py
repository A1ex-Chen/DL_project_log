def get(self):
    """Convenience function for accessing box coordinates.

    Returns:
      a tensor with shape [N, 4] representing box coordinates.
    """
    return self.get_field('boxes')
