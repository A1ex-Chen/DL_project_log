def cache_labels(self, path=Path('./labels.cache'), prefix=''):
    x = {}
    nm, nf, ne, nc = 0, 0, 0, 0
    pbar = tqdm(zip(self.img_files, self.label_files), desc=
        'Scanning images', total=len(self.img_files))
    for i, (im_file, lb_file) in enumerate(pbar):
        try:
            im = Image.open(im_file)
            im.verify()
            shape = exif_size(im)
            segments = []
            assert (shape[0] > 9) & (shape[1] > 9
                ), f'image size {shape} <10 pixels'
            assert im.format.lower(
                ) in img_formats, f'invalid image format {im.format}'
            if os.path.isfile(lb_file):
                nf += 1
                with open(lb_file, 'r') as f:
                    l = [x.split() for x in f.read().strip().splitlines()]
                    if any([(len(x) > 8) for x in l]):
                        classes = np.array([x[0] for x in l], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).
                            reshape(-1, 2) for x in l]
                        l = np.concatenate((classes.reshape(-1, 1),
                            segments2boxes(segments)), 1)
                    l = np.array(l, dtype=np.float32)
                if len(l):
                    assert l.shape[1] == 5, 'labels require 5 columns each'
                    assert (l >= 0).all(), 'negative labels'
                    assert (l[:, 1:] <= 1).all(
                        ), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(l, axis=0).shape[0] == l.shape[0
                        ], 'duplicate labels'
                else:
                    ne += 1
                    l = np.zeros((0, 5), dtype=np.float32)
            else:
                nm += 1
                l = np.zeros((0, 5), dtype=np.float32)
            x[im_file] = [l, shape, segments]
        except Exception as e:
            nc += 1
            print(
                f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
                )
        pbar.desc = (
            f"{prefix}Scanning '{path.parent / path.stem}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            )
    pbar.close()
    if nf == 0:
        print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
    x['hash'] = get_hash(self.label_files + self.img_files)
    x['results'] = nf, nm, ne, nc, i + 1
    x['version'] = 0.1
    torch.save(x, path)
    logging.info(f'{prefix}New cache created: {path}')
    return x
