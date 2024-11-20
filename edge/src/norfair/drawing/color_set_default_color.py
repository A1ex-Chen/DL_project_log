@classmethod
def set_default_color(cls, color: ColorLike):
    """
        Selects the default color of `choose_color` when hashable is None.

        Parameters
        ----------
        color : ColorLike
            The new default color.
        """
    cls._default_color = parse_color(color)
