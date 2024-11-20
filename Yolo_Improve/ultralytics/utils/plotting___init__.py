def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf',
    pil=False, example='abc'):
    """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
    non_ascii = not is_ascii(example)
    input_is_pil = isinstance(im, Image.Image)
    self.pil = pil or non_ascii or input_is_pil
    self.lw = line_width or max(round(sum(im.size if input_is_pil else im.
        shape) / 2 * 0.003), 2)
    if self.pil:
        self.im = im if input_is_pil else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)
        try:
            font = check_font('Arial.Unicode.ttf' if non_ascii else font)
            size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
            self.font = ImageFont.truetype(str(font), size)
        except Exception:
            self.font = ImageFont.load_default()
        if check_version(pil_version, '9.2.0'):
            self.font.getsize = lambda x: self.font.getbbox(x)[2:4]
    else:
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images.'
        self.im = im if im.flags.writeable else im.copy()
        self.tf = max(self.lw - 1, 1)
        self.sf = self.lw / 3
    self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 
        12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1,
        2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0,
        0, 16, 16, 16, 16, 16, 16, 16]]
    self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
        0, 9, 9, 9, 9, 9, 9]]
    self.dark_colors = {(235, 219, 11), (243, 243, 243), (183, 223, 0), (
        221, 111, 255), (0, 237, 204), (68, 243, 0), (255, 255, 0), (179, 
        255, 1), (11, 255, 162)}
    self.light_colors = {(255, 42, 4), (79, 68, 255), (255, 0, 189), (255, 
        180, 0), (186, 0, 221), (0, 192, 38), (255, 36, 125), (104, 0, 123),
        (108, 27, 255), (47, 109, 252), (104, 31, 17)}
