@classmethod
def line(cls, frame: np.ndarray, start: Tuple[int, int], end: Tuple[int,
    int], color: ColorType=Color.black, thickness: int=1) ->np.ndarray:
    """
        Draw a line.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        start : Tuple[int, int]
            Starting point.
        end : Tuple[int, int]
            End point.
        color : ColorType, optional
            Line color.
        thickness : int, optional
            Line width.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    return cv2.line(frame, pt1=start, pt2=end, color=color, thickness=thickness
        )
