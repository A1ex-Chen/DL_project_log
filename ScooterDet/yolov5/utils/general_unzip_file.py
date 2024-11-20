def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)
