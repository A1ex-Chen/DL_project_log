def visualize_frames(self, video_path: str, save_path: str, **kwargs) ->None:
    """Visualize bounding boxes on a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            None
        """

    def generator():
        movie_iterator = MovieIterator(video_path)
        for frame_idx, frame in zip(self.index, movie_iterator):
            img_ = self.visualize_frame(frame_idx, frame)
            yield img_
    input_framerate = get_fps(video_path)
    make_video(generator(), save_path, input_framerate=input_framerate, **
        kwargs)
