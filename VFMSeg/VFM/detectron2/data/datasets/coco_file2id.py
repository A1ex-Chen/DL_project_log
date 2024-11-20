def file2id(folder_path, file_path):
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    image_id = os.path.splitext(image_id)[0]
    return image_id
