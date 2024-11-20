def __next__(self):
    self.count += 1
    img0 = self.imgs.copy()
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        raise StopIteration
    img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[
        0] for x in img0]
    img = np.stack(img, 0)
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
    img = np.ascontiguousarray(img)
    return self.sources, img, img0, None
