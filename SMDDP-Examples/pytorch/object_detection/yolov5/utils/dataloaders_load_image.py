def load_image(self, i):
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        if fn.exists():
            im = np.load(fn)
        else:
            im = cv2.imread(f)
            assert im is not None, f'Image Not Found {f}'
        h0, w0 = im.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = (cv2.INTER_LINEAR if self.augment or r > 1 else cv2.
                INTER_AREA)
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=
                interp)
        return im, (h0, w0), im.shape[:2]
    return self.ims[i], self.im_hw0[i], self.im_hw[i]
