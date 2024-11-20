@staticmethod
def font_check(font='./yolov6/utils/Arial.ttf', size=10):
    assert osp.exists(font), f'font path not exists: {font}'
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name,
            size)
    except Exception as e:
        return ImageFont.truetype(str(font), size)
