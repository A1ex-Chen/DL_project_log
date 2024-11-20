def update(self, i, cap, stream):
    n, f = 0, self.frames[i]
    while cap.isOpened() and n < f:
        n += 1
        cap.grab()
        if n % self.vid_stride == 0:
            success, im = cap.retrieve()
            if success:
                self.imgs[i] = im
            else:
                LOGGER.warning(
                    'WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.'
                    )
                self.imgs[i] = np.zeros_like(self.imgs[i])
                cap.open(stream)
        time.sleep(0.0)
