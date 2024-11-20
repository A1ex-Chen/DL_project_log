def __init__(self):
    hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
        '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
        '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
        'FF95C8', 'FF37C7')
    self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
    self.n = len(self.palette)
