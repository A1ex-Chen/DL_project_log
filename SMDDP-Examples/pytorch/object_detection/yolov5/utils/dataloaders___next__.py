def __next__(self):
    self.count += 1
    if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'
        ):
        cv2.destroyAllWindows()
        raise StopIteration
    img0 = self.imgs.copy()
    img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and
        self.auto)[0] for x in img0]
    img = np.stack(img, 0)
    img = img[..., ::-1].transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(img)
    return self.sources, img, img0, None, ''
