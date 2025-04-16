def _load_data(basedir, factor=None, width=None, height=None, load_imgs=
    True, crop_size=0, load_colmap_poses=True):
    if load_colmap_poses:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
    img_folder = 'images'
    crop_ratio = 1
    focal_crop_factor = 1
    if crop_size != 0:
        img_folder = 'images_cropped'
        crop_dir = os.path.join(basedir, 'images_cropped')
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
        for f in sorted(os.listdir(os.path.join(basedir, 'images'))):
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
                image = imageio.imread(os.path.join(basedir, 'images', f))
                crop_size_H = crop_size
                H, W, _ = image.shape
                crop_size_W = int(crop_size_H * W / H)
                image_cropped = image[crop_size_H:H - crop_size_H,
                    crop_size_W:W - crop_size_W]
                save_path = os.path.join(crop_dir, f)
                im = Image.fromarray(image_cropped)
                im = im.resize((W, H))
                im.save(save_path)
        crop_ratio = crop_size_H / H
        print('=======images cropped=======')
        focal_crop_factor = (H - 2 * crop_size_H) / H
    img0 = [os.path.join(basedir, img_folder, f) for f in sorted(os.listdir
        (os.path.join(basedir, img_folder))) if f.endswith('JPG') or f.
        endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor], img_folder=img_folder)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]], img_folder=img_folder)
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]], img_folder=img_folder)
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    imgdir = os.path.join(basedir, img_folder + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
        f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    sh = imageio.imread(imgfiles[0]).shape
    if load_colmap_poses:
        if poses.shape[-1] != len(imgfiles):
            print('Mismatch between imgs {} and poses {} !!!!'.format(len(
                imgfiles), poses.shape[-1]))
            return
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    imgs = imgs = [(imread(f)[..., :3] / 255.0) for f in imgfiles]
    imgs = np.stack(imgs, -1)
    if load_colmap_poses:
        print('Loaded image data', imgs.shape, poses[:, -1, 0])
    else:
        print('Loaded image data', imgs.shape)
        poses = None
        bds = None
    imgnames = [f for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or
        f.endswith('jpg') or f.endswith('png')]
    return poses, bds, imgs, imgnames, crop_ratio, focal_crop_factor
