@classmethod
def cross(cls, frame: np.ndarray, center: Tuple[int, int], radius: int,
    color: ColorType, thickness: int) ->np.ndarray:
    """
        Draw a cross

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        center : Tuple[int, int]
            Center of the cross.
        radius : int
            Size or radius of the cross.
        color : Color
            Color of the lines.
        thickness : int
            Thickness of the lines.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    middle_x, middle_y = center
    left, top = center - radius
    right, bottom = center + radius
    frame = cls.line(frame, start=(middle_x, top), end=(middle_x, bottom),
        color=color, thickness=thickness)
    frame = cls.line(frame, start=(left, middle_y), end=(right, middle_y),
        color=color, thickness=thickness)
    return frame
