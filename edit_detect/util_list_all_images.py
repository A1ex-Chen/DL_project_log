def list_all_images(root: Union[str, os.PathLike, pathlib.PurePath],
    image_exts: List[str]=['png', 'jpg', 'jpeg', 'webp'],
    ext_case_sensitive: bool=False):
    img_ls_all: List[str] = glob.glob(f'{str(root)}/**', recursive=True)
    img_ls_filtered: List[str] = []
    for img in img_ls_all:
        img_path: str = str(img)
        img_path_ext: str = img_path.split('.')[-1]
        for image_ext in image_exts:
            if ext_case_sensitive:
                if img_path_ext == str(image_ext):
                    img_ls_filtered.append(img_path)
            elif img_path_ext.lower() == str(image_ext).lower():
                img_ls_filtered.append(img_path)
    return img_ls_filtered
