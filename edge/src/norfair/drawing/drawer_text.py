@classmethod
def text(cls, frame: np.ndarray, text: str, position: Tuple[int, int], size:
    Optional[float]=None, color: Optional[ColorType]=None, thickness:
    Optional[int]=None, shadow: bool=True, shadow_color: ColorType=Color.
    black, shadow_offset: int=1) ->np.ndarray:
    """
        Draw text

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        text : str
            The text to be written.
        position : Tuple[int, int]
            Position of the bottom-left corner of the text.
            This value is adjusted considering the thickness automatically.
        size : Optional[float], optional
            Scale of the font, by default chooses a sensible value is picked based on the size of the frame.
        color : Optional[ColorType], optional
            Color of the text, by default is black.
        thickness : Optional[int], optional
            Thickness of the lines, by default a sensible value is picked based on the size.
        shadow : bool, optional
            If True, a shadow of the text is added which improves legibility.
        shadow_color : Color, optional
            Color of the shadow.
        shadow_offset : int, optional
            Offset of the shadow.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    if size is None:
        size = min(max(max(frame.shape) / 4000, 0.5), 1.5)
    if thickness is None:
        thickness = int(round(size) + 1)
    if thickness is None and size is not None:
        thickness = int(round(size) + 1)
    anchor = position[0] + thickness // 2, position[1] - thickness // 2
    if shadow:
        frame = cv2.putText(frame, text, (anchor[0] + shadow_offset, anchor
            [1] + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, size,
            shadow_color, thickness, cv2.LINE_AA)
    return cv2.putText(frame, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, size,
        color, thickness, cv2.LINE_AA)
