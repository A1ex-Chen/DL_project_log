def show(self, p=''):
    """Display an image in a window using OpenCV imshow()."""
    im = self.plotted_img
    if platform.system() == 'Linux' and p not in self.windows:
        self.windows.append(p)
        cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(p, im.shape[1], im.shape[0])
    cv2.imshow(p, im)
    cv2.waitKey(300 if self.dataset.mode == 'image' else 1)
