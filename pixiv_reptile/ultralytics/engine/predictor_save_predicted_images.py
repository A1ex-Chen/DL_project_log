def save_predicted_images(self, save_path='', frame=0):
    """Save video predictions as mp4 at specified path."""
    im = self.plotted_img
    if self.dataset.mode in {'stream', 'video'}:
        fps = self.dataset.fps if self.dataset.mode == 'video' else 30
        frames_path = f"{save_path.split('.', 1)[0]}_frames/"
        if save_path not in self.vid_writer:
            if self.args.save_frames:
                Path(frames_path).mkdir(parents=True, exist_ok=True)
            suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2'
                ) if WINDOWS else ('.avi', 'MJPG')
            self.vid_writer[save_path] = cv2.VideoWriter(filename=str(Path(
                save_path).with_suffix(suffix)), fourcc=cv2.
                VideoWriter_fourcc(*fourcc), fps=fps, frameSize=(im.shape[1
                ], im.shape[0]))
        self.vid_writer[save_path].write(im)
        if self.args.save_frames:
            cv2.imwrite(f'{frames_path}{frame}.jpg', im)
    else:
        cv2.imwrite(save_path, im)
