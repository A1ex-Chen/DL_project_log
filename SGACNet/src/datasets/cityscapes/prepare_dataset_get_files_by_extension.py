def get_files_by_extension(path, extension='.png', flat_structure=False,
    recursive=False, follow_links=True):
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))
    if flat_structure:
        filelist = []
    else:
        filelist = {}
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True, followlinks=
        follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend(os.path.join(root, f) for f in filenames)
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))
