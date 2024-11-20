def generator():
    movie_iterator = MovieIterator(video_path)
    for frame_idx, frame in zip(self.index, movie_iterator):
        img_ = self.visualize_frame(frame_idx, frame)
        yield img_
