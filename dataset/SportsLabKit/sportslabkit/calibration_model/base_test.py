def test(self):
    import cv2
    from sportslabkit.utils.utils import get_git_root
    git_root = get_git_root()
    im_path = git_root / 'data' / 'samples' / 'ney.jpeg'
    imgs = [str(im_path), im_path,
        'https://ultralytics.com/images/zidane.jpg', cv2.imread(str(im_path
        ))[:, :, ::-1], Image.open(str(im_path)), np.zeros((320, 640, 3))]
    results = self(imgs)
    print(results)
    for img in imgs:
        results = self(img)
        print(results)
