def cv_imread(filePath):
    img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    plt.imshow(img)
    return img
