def process_img(img, new_shape=(640, 480), isRGB=True):
    img = cv2.resize(img, new_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    return img
