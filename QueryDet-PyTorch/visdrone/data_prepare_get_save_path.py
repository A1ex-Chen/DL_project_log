def get_save_path(img_path, index):
    name = img_path.split('.')[0]
    return name + '_' + str(index) + '.jpg'
