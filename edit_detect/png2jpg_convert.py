def convert(src: str, dest: str):
    im = Image.open(src)
    rgb_im = im.convert('RGB')
    rgb_im.save(dest)
