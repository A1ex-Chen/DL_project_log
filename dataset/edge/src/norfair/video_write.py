def write(self, frame: np.ndarray) ->int:
    """
        Write one frame to the output video.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to write to file.

        Returns
        -------
        int
            _description_
        """
    if self.output_video is None:
        output_file_path = self.get_output_file_path()
        fourcc = cv2.VideoWriter_fourcc(*self.get_codec_fourcc(
            output_file_path))
        output_size = frame.shape[1], frame.shape[0]
        self.output_video = cv2.VideoWriter(output_file_path, fourcc, self.
            output_fps, output_size)
    self.output_video.write(frame)
    return cv2.waitKey(1)
