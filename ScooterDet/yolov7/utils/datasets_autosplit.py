def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)
    files = sum([list(path.rglob(f'*.{img_ext}')) for img_ext in
        img_formats], [])
    n = len(files)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    [(path / x).unlink() for x in txt if (path / x).exists()]
    print(f'Autosplitting images from {path}' + 
        ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')
