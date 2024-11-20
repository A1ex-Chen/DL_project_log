@staticmethod
def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
        '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
        '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
        'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color
