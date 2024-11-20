@classmethod
def rectangle(cls, frame: np.ndarray, points: Sequence[Tuple[int, int]],
    color: Optional[ColorType]=None, thickness: Optional[int]=None
    ) ->np.ndarray:
    """
        Draw a rectangle

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        points : Sequence[Tuple[int, int]]
            Points describing the rectangle in the format `[[x0, y0], [x1, y1]]`.
        color : Optional[ColorType], optional
            Color of the lines, by default Black.
        thickness : Optional[int], optional
            Thickness of the lines, by default 1.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    frame = cv2.rectangle(frame, tuple(points[0]), tuple(points[1]), color=
        color, thickness=thickness)
    return frame
