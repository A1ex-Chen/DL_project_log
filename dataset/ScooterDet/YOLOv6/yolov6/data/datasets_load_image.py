def load_image(self, index, shrink_size=None):
    """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
    path = self.img_paths[index]
    if self.cache_ram and self.imgs[index] is not None:
        im = self.imgs[index]
        return self.imgs[index], self.imgs_hw0[index], self.imgs_hw[index]
    else:
        try:
            im = cv2.imread(path)
            assert im is not None, f'opencv cannot read image correctly or {path} not exists'
        except Exception as e:
            print(e)
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f'Image Not Found {path}, workdir: {os.getcwd()}'
        h0, w0 = im.shape[:2]
        if self.specific_shape:
            ratio = min(self.target_width / w0, self.target_height / h0)
        elif shrink_size:
            ratio = (self.img_size - shrink_size) / max(h0, w0)
        else:
            ratio = self.img_size / max(h0, w0)
        if ratio != 1:
            im = cv2.resize(im, (int(w0 * ratio), int(h0 * ratio)),
                interpolation=cv2.INTER_AREA if ratio < 1 and not self.
                augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]
