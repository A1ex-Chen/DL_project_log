def convert(dir, width):
    file_list = os.listdir(dir)
    for filename in file_list:
        path = ''
        path = dir + filename
        im = Image.open(path)
        if im.mode == 'P' or im.mode == 'RGBA':
            im = im.convert('RGB')
        out = im.resize((width, width), Image.ANTIALIAS)
        print('%s has been resized!' % filename)
        tmp_dir = 'C:\\Users\\10696\\Desktop\\photos'
        resize_img_path = os.path.join(tmp_dir, filename)
        out.save(resize_img_path)
