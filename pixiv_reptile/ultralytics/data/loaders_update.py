def update(self, i, cap, stream):
    """Read stream `i` frames in daemon thread."""
    n, f = 0, self.frames[i]
    while self.running and cap.isOpened() and n < f - 1:
        if len(self.imgs[i]) < 30:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if not success:
                    im = np.zeros(self.shape[i], dtype=np.uint8)
                    LOGGER.warning(
                        'WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.'
                        )
                    cap.open(stream)
                if self.buffer:
                    self.imgs[i].append(im)
                else:
                    self.imgs[i] = [im]
        else:
            time.sleep(0.01)
