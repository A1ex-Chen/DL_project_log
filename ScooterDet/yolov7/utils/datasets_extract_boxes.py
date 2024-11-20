def extract_boxes(path='../coco/'):
    path = Path(path)
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir(
        ) else None
    files = list(path.rglob('*.*'))
    n = len(files)
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            im = cv2.imread(str(im_file))[..., ::-1]
            h, w = im.shape[:2]
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().
                        splitlines()], dtype=np.float32)
                for j, x in enumerate(lb):
                    c = int(x[0])
                    f = (path / 'classifier' / f'{c}' /
                        f'{path.stem}_{im_file.stem}_{j}.jpg')
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)
                    b = x[1:] * [w, h, w, h]
                    b[2:] = b[2:] * 1.2 + 3
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]
                        ), f'box failure in {f}'
