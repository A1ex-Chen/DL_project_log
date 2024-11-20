def load_image(self, i, rect_mode=True):
    """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        if fn.exists():
            try:
                im = np.load(fn)
            except Exception as e:
                LOGGER.warning(
                    f'{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}'
                    )
                Path(fn).unlink(missing_ok=True)
                im = cv2.imread(f)
        else:
            im = cv2.imread(f)
        if im is None:
            raise FileNotFoundError(f'Image Not Found {f}')
        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 *
                    r), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not h0 == w0 == self.imgsz:
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2
                .INTER_LINEAR)
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0
                ), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != 'ram':
                    self.ims[j], self.im_hw0[j], self.im_hw[j
                        ] = None, None, None
        return im, (h0, w0), im.shape[:2]
    return self.ims[i], self.im_hw0[i], self.im_hw[i]
