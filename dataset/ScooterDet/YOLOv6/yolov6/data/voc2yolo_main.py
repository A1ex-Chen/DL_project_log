def main(args):
    voc_path = args.voc_path
    for year, image_set in (('2012', 'train'), ('2012', 'val'), ('2007',
        'train'), ('2007', 'val'), ('2007', 'test')):
        imgs_path = os.path.join(voc_path, 'images', f'{image_set}')
        lbs_path = os.path.join(voc_path, 'labels', f'{image_set}')
        try:
            with open(os.path.join(voc_path,
                f'VOC{year}/ImageSets/Main/{image_set}.txt'), 'r') as f:
                image_ids = f.read().strip().split()
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)
            if not os.path.exists(lbs_path):
                os.makedirs(lbs_path)
            for id in tqdm(image_ids, desc=f'{image_set}{year}'):
                f = os.path.join(voc_path, f'VOC{year}/JPEGImages/{id}.jpg')
                lb_path = os.path.join(lbs_path, f'{id}.txt')
                convert_label(voc_path, lb_path, year, id)
                if os.path.exists(f):
                    shutil.move(f, imgs_path)
        except Exception as e:
            print(f'[Warning]: {e} {year}{image_set} convert fail!')
    gen_voc07_12(voc_path)
