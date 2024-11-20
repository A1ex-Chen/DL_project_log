def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0),
    annotated_only=False):
    """Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in
        IMG_FORMATS)
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()
    print(f'Autosplitting images from {path}' + 
        ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')
