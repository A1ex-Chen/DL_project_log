@classmethod
def circle(cls, frame: np.ndarray, position: Tuple[int, int], radius:
    Optional[int]=None, thickness: Optional[int]=None, color: ColorType=None
    ) ->np.ndarray:
    """
        Draw a circle.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        position : Tuple[int, int]
            Position of the point. This will become the center of the circle.
        radius : Optional[int], optional
            Radius of the circle.
        thickness : Optional[int], optional
            Thickness or width of the line.
        color : Color, optional
            A tuple of ints describing the BGR color `(0, 0, 255)`.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    if radius is None:
        radius = int(max(max(frame.shape) * 0.005, 1))
    if thickness is None:
        thickness = radius - 1
    return cv2.circle(frame, position, radius=radius, color=color,
        thickness=thickness)
