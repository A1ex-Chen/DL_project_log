def __next__(self):
    if self.frame_number <= self.length:
        frame_path = os.path.join(self.input_path, self.image_directory, 
            str(self.frame_number).zfill(6) + self.image_extension)
        self.frame_number += 1
        return cv2.imread(frame_path)
    raise StopIteration()
