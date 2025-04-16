def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        path = self.img_files[index]
        img = cv2.imread(path)
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = (cv2.INTER_AREA if r < 1 and not self.augment else cv2
                .INTER_LINEAR)
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation
                =interp)
        return img, (h0, w0), img.shape[:2]
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]
