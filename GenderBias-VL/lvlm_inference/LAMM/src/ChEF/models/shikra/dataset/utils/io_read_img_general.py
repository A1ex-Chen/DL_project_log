def read_img_general(img_path):
    if 's3://' in img_path:
        cv_img = read_img_ceph(img_path)
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        return Image.open(img_path).convert('RGB')
