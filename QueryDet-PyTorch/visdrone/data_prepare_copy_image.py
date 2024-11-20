def copy_image(img_root, img_path, new_img_root):
    img = cv2.imread(os.path.join(img_root, img_path))
    h, w, c = img.shape
    cv2.imwrite(os.path.join(new_img_root, img_path), img)
    return h, w
