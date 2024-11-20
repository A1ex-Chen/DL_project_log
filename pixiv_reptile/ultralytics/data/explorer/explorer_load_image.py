def load_image(self, i: int) ->Union[Tuple[np.ndarray, Tuple[int, int],
    Tuple[int, int]], Tuple[None, None, None]]:
    """Loads 1 image from dataset index 'i' without any resize ops."""
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        if fn.exists():
            im = np.load(fn)
        else:
            im = cv2.imread(f)
            if im is None:
                raise FileNotFoundError(f'Image Not Found {f}')
        h0, w0 = im.shape[:2]
        return im, (h0, w0), im.shape[:2]
    return self.ims[i], self.im_hw0[i], self.im_hw[i]
