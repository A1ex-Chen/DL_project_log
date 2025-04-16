def load_img_dirs(img_dirs):
    img_list = []
    for img_path in os.listdir(img_dirs):
        img_path = os.path.join(img_dirs, img_path)
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        img_list.append(img)
        del img
    if len(img_list) == 0:
        return None
    img_list = torch.concat(img_list, dim=0)
    return img_list
