def reset_image(self, img):
    img = img.astype('uint8')
    self.ax.imshow(img, extent=(0, self.width, self.height, 0),
        interpolation='nearest')
