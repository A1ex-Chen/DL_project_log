def img2label_paths(img_paths):
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt') for x in
        img_paths]
