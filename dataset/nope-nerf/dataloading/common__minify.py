def _minify(basedir, factors=[], resolutions=[], img_folder='images'):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, img_folder + '_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, img_folder + '_{}x{}'.format(r[1], r[0])
            )
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    from shutil import copy
    from subprocess import check_output
    imgdir = os.path.join(basedir, img_folder)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg',
        'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    wd = os.getcwd()
    for r in (factors + resolutions):
        if isinstance(r, int):
            name = img_folder + '_{}'.format(r)
            resizearg = '{}%'.format(100.0 / r)
        else:
            name = img_folder + '_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png',
            '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
