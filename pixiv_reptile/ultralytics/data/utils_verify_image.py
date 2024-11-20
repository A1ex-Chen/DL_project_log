def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args
    nf, nc, msg = 0, 0, ''
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = shape[1], shape[0]
        assert (shape[0] > 9) & (shape[1] > 9
            ), f'image size {shape} <10 pixels'
        assert im.format.lower(
            ) in IMG_FORMATS, f'Invalid image format {im.format}. {FORMATS_HELP_MSG}'
        if im.format.lower() in {'jpg', 'jpeg'}:
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file,
                        'JPEG', subsampling=0, quality=100)
                    msg = (
                        f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
                        )
        nf = 1
    except Exception as e:
        nc = 1
        msg = (
            f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}')
    return (im_file, cls), nf, nc, msg
