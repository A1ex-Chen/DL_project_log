def update(self, index, cap):
    n = 0
    while cap.isOpened():
        n += 1
        cap.grab()
        if n == 4:
            success, im = cap.retrieve()
            self.imgs[index] = im if success else self.imgs[index] * 0
            n = 0
        time.sleep(1 / self.fps)
