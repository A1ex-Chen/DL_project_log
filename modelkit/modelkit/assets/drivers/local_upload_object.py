def upload_object(self, file_path, object_name):
    object_path = os.path.join(self.bucket, *object_name.split('/'))
    object_dir, _ = os.path.split(object_path)
    if os.path.isfile(object_path):
        os.remove(object_path)
    if os.path.isdir(object_path):
        shutil.rmtree(object_path)
    if os.path.isfile(object_dir):
        os.remove(object_dir)
    os.makedirs(object_dir, exist_ok=True)
    with open(file_path, 'rb') as fsrc:
        with open(object_path, 'xb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
