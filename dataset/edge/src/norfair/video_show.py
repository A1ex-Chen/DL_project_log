def show(self, frame: np.ndarray, downsample_ratio: float=1.0) ->int:
    """
        Display a frame through a GUI. Usually used inside a video inference loop to show the output video.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to be displayed.
        downsample_ratio : float, optional
            How much to downsample the frame being show.

            Useful when streaming the GUI video display through a slow internet connection using something like X11 forwarding on an ssh connection.

        Returns
        -------
        int
            _description_
        """
    if downsample_ratio != 1.0:
        frame = cv2.resize(frame, (frame.shape[1] // downsample_ratio, 
            frame.shape[0] // downsample_ratio))
    cv2.imshow('Output', frame)
    return cv2.waitKey(1)
