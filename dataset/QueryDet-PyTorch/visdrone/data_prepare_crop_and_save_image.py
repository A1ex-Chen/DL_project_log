def crop_and_save_image(img_root, img_path, new_img_root):
    img = cv2.imread(os.path.join(img_root, img_path))
    h, w, c = img.shape
    _y = h // 2
    _x = w // 2
    img0 = img[:_y, :_x, :]
    img1 = img[:_y, _x:, :]
    img2 = img[_y:, :_x, :]
    img3 = img[_y:, _x:, :]
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 0)), img0)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 1)), img1)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 2)), img2)
    cv2.imwrite(os.path.join(new_img_root, get_save_path(img_path, 3)), img3)
    return h, w, _y, _x
