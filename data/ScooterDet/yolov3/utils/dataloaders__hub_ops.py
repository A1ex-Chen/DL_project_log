def _hub_ops(self, f, max_dim=1920):
    f_new = self.im_dir / Path(f).name
    try:
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)
        if r < 1.0:
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new, 'JPEG', quality=50, optimize=True)
    except Exception as e:
        LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)
        if r < 1.0:
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)),
                interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new), im)
