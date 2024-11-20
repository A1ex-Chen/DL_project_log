def get_img_files(self, img_path):
    """Read image files."""
    try:
        f = []
        for p in (img_path if isinstance(img_path, list) else [img_path]):
            p = Path(p)
            if p.is_dir():
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [(x.replace('./', parent) if x.startswith('./') else
                        x) for x in t]
            else:
                raise FileNotFoundError(f'{self.prefix}{p} does not exist')
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')
            [-1].lower() in IMG_FORMATS)
        assert im_files, f'{self.prefix}No images found'
    except Exception as e:
        raise FileNotFoundError(
            f'{self.prefix}Error loading data from {img_path}') from e
    if self.fraction < 1:
        im_files = im_files[:round(len(im_files) * self.fraction)]
    return im_files
