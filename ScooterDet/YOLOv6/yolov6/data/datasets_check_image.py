@staticmethod
def check_image(im_file):
    """Verify an image."""
    nc, msg = 0, ''
    try:
        im = Image.open(im_file)
        im.verify()
        im = Image.open(im_file)
        shape = im.height, im.width
        try:
            im_exif = im._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = shape[1], shape[0]
        except:
            im_exif = None
        assert (shape[0] > 9) & (shape[1] > 9
            ), f'image size {shape} <10 pixels'
        assert im.format.lower(
            ) in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file,
                        'JPEG', subsampling=0, quality=100)
                    msg += (
                        f'WARNING: {im_file}: corrupt JPEG restored and saved')
        return im_file, shape, nc, msg
    except Exception as e:
        nc = 1
        msg = f'WARNING: {im_file}: ignoring corrupt image: {e}'
        return im_file, None, nc, msg
