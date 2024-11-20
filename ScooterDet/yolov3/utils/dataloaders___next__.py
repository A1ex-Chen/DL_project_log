def __next__(self):
    self.count += 1
    if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'
        ):
        cv2.destroyAllWindows()
        raise StopIteration
    im0 = self.imgs.copy()
    if self.transforms:
        im = np.stack([self.transforms(x) for x in im0])
    else:
        im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto
            =self.auto)[0] for x in im0])
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
    return self.sources, im, im0, None, ''
