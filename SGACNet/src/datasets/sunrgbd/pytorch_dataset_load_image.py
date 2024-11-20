def load_image(self, idx):
    if self.camera is None:
        img_dir = self.img_dir[self._split]['list']
    else:
        img_dir = self.img_dir[self._split]['dict'][self.camera]
    fp = os.path.join(self._data_dir, img_dir[idx])
    image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
