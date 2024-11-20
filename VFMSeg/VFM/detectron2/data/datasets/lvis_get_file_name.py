def get_file_name(img_root, img_dict):
    split_folder, file_name = img_dict['coco_url'].split('/')[-2:]
    return os.path.join(img_root + split_folder, file_name)
