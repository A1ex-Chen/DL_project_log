def load_image(self, i):
    """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        if fn.exists():
            im = np.load(fn)
        else:
            im = cv2.imread(f)
            if im is None:
                raise FileNotFoundError(f'Image Not Found {f}')
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = (cv2.INTER_LINEAR if self.augment or r > 1 else cv2.
                INTER_AREA)
            im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(
                math.ceil(h0 * r), self.imgsz)), interpolation=interp)
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0
                ), im.shape[:2]
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]
    return self.ims[i], self.im_hw0[i], self.im_hw[i]
