def unzip(zip_path: str, dest_dir: str) ->None:
    with ZipFile(zip_path, 'r') as zipObj:
        zipObj.extractall(dest_dir)
