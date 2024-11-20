def make_dataset(directory: str, class_to_idx: Dict[str, int], extensions:
    Optional[Tuple[str, ...]]=None, is_valid_file: Optional[Callable[[str],
    bool]]=None, img_list: Optional[str]=None) ->List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            'Both extensions and is_valid_file cannot be None or not None at the same time'
            )
    if extensions is not None:

        def is_valid_file(x: str) ->bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...],
                extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    if img_list is not None:
        print('========= img_list is not None =========')
        class_to_imgPath = dict()
        with open(img_list, 'r') as fin:
            for one_line in fin.readlines():
                path = one_line.strip()
                target_class = path.split(directory)[-1].lstrip('/').split('/'
                    )[0]
                assert target_class in class_to_idx, 'ERROR! target_class [{}] is not in class_to_idx [{}]'.format(
                    target_class, class_to_idx)
                if target_class not in class_to_imgPath:
                    class_to_imgPath[target_class] = list()
                class_to_imgPath[target_class].append(path)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            all_image_paths = sorted(class_to_imgPath[target_class])
            for path in all_image_paths:
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    else:
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)
                ):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
    return instances
