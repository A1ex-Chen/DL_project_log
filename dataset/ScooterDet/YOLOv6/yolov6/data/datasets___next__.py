def __next__(self):
    if self.count == self.nf:
        raise StopIteration
    path = self.files[self.count]
    if self.checkext(path) == 'video':
        self.type = 'video'
        ret_val, img = self.cap.read()
        while not ret_val:
            self.count += 1
            self.cap.release()
            if self.count == self.nf:
                raise StopIteration
            path = self.files[self.count]
            self.add_video(path)
            ret_val, img = self.cap.read()
    else:
        self.count += 1
        img = cv2.imread(path)
    return img, path, self.cap
