def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf',
    pil=False, example='abc'):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
    non_ascii = not is_ascii(example)
    self.pil = pil or non_ascii
    if self.pil:
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)
        self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else
            font, size=font_size or max(round(sum(self.im.size) / 2 * 0.035
            ), 12))
    else:
        self.im = im
    self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
