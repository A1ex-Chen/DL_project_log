@classmethod
def alpha_blend(cls, frame1: np.ndarray, frame2: np.ndarray, alpha: float=
    0.5, beta: Optional[float]=None, gamma: float=0) ->np.ndarray:
    """
        Blend 2 frame as a wheigthted sum.

        Parameters
        ----------
        frame1 : np.ndarray
            An OpenCV frame.
        frame2 : np.ndarray
            An OpenCV frame.
        alpha : float, optional
            Weight of frame1.
        beta : Optional[float], optional
            Weight of frame2, by default `1 - alpha`
        gamma : float, optional
            Scalar to add to the sum.

        Returns
        -------
        np.ndarray
            The resulting frame.
        """
    if beta is None:
        beta = 1 - alpha
    return cv2.addWeighted(src1=frame1, src2=frame2, alpha=alpha, beta=beta,
        gamma=gamma)
