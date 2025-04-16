def is_all_black(self, img):
    image = np.array(img)
    return np.all(image == [0, 0, 0])
