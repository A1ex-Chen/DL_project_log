def create_folder(path='./new'):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
