def __repr__(self):
    """Return string representation of object."""
    format_string = f'{self.__class__.__name__}('
    for t in self.transforms:
        format_string += '\n'
        format_string += f'    {t}'
    format_string += '\n)'
    return format_string
