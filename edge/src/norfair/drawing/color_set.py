@classmethod
def set(cls, palette: Union[str, Iterable[ColorLike]]):
    """
        Selects a color palette.

        Parameters
        ----------
        palette : Union[str, Iterable[ColorLike]]
            can be either
            - the name of one of the predefined palettes `tab10`, `tab20`, or `colorblind`
            - a list of ColorLike objects that can be parsed by [`parse_color`][norfair.drawing.color.parse_color]
        """
    if isinstance(palette, str):
        try:
            cls._colors = PALETTES[palette]
        except KeyError as e:
            raise ValueError(
                f"Invalid palette name '{palette}', valid values are {PALETTES.keys()}"
                ) from e
    else:
        colors = []
        for c in palette:
            colors.append(parse_color(c))
        cls._colors = colors
