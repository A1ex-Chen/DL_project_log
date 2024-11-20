def __init__(self, root: str, transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None, loader: Callable[[str], Any]
    =default_loader, is_valid_file: Optional[Callable[[str], bool]]=None,
    img_list: Optional[str]=None):
    super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if 
        is_valid_file is None else None, transform=transform,
        target_transform=target_transform, is_valid_file=is_valid_file,
        img_list=img_list)
    self.imgs = self.samples
