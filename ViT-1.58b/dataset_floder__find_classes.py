def _find_classes(self, dir: str, path_file: Optional[str]=None) ->Tuple[
    List[str], Dict[str, int]]:
    classes = []
    if path_file is None:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        with open(path_file, 'r') as fin:
            for line in fin.readlines():
                line = line.strip()
                cls = os.path.expanduser(line).split(dir)[-1].lstrip('/'
                    ).split('/')[0]
                if cls not in classes:
                    classes.append(cls)
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
