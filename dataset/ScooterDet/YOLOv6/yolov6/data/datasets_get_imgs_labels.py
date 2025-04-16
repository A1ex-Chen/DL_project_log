def get_imgs_labels(self, img_dirs):
    if not isinstance(img_dirs, list):
        img_dirs = [img_dirs]
    valid_img_record = osp.join(osp.dirname(img_dirs[0]), '.' + osp.
        basename(img_dirs[0]) + '_cache.json')
    NUM_THREADS = min(8, os.cpu_count())
    img_paths = []
    for img_dir in img_dirs:
        assert osp.exists(img_dir), f'{img_dir} is an invalid directory path!'
        img_paths += glob.glob(osp.join(img_dir, '**/*'), recursive=True)
    img_paths = sorted(p for p in img_paths if p.split('.')[-1].lower() in
        IMG_FORMATS and os.path.isfile(p))
    assert img_paths, f'No images found in {img_dir}.'
    img_hash = self.get_hash(img_paths)
    LOGGER.info(f'img record infomation path is:{valid_img_record}')
    if osp.exists(valid_img_record):
        with open(valid_img_record, 'r') as f:
            cache_info = json.load(f)
            if 'image_hash' in cache_info and cache_info['image_hash'
                ] == img_hash:
                img_info = cache_info['information']
            else:
                self.check_images = True
    else:
        self.check_images = True
    if self.check_images and self.main_process:
        img_info = {}
        nc, msgs = 0, []
        LOGGER.info(
            f'{self.task}: Checking formats of images with {NUM_THREADS} process(es): '
            )
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(TrainValDataset.check_image, img_paths),
                total=len(img_paths))
            for img_path, shape_per_img, nc_per_img, msg in pbar:
                if nc_per_img == 0:
                    img_info[img_path] = {'shape': shape_per_img}
                nc += nc_per_img
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{nc} image(s) corrupted'
        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        cache_info = {'information': img_info, 'image_hash': img_hash}
        with open(valid_img_record, 'w') as f:
            json.dump(cache_info, f)
    img_paths = list(img_info.keys())
    label_paths = img2label_paths(img_paths)
    assert label_paths, f'No labels found.'
    label_hash = self.get_hash(label_paths)
    if 'label_hash' not in cache_info or cache_info['label_hash'
        ] != label_hash:
        self.check_labels = True
    if self.check_labels:
        cache_info['label_hash'] = label_hash
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        LOGGER.info(
            f'{self.task}: Checking formats of labels with {NUM_THREADS} process(es): '
            )
        with Pool(NUM_THREADS) as pool:
            pbar = pool.imap(TrainValDataset.check_label_files, zip(
                img_paths, label_paths))
            pbar = tqdm(pbar, total=len(label_paths)
                ) if self.main_process else pbar
            for img_path, labels_per_file, nc_per_file, nm_per_file, nf_per_file, ne_per_file, msg in pbar:
                if nc_per_file == 0:
                    img_info[img_path]['labels'] = labels_per_file
                else:
                    img_info.pop(img_path)
                nc += nc_per_file
                nm += nm_per_file
                nf += nf_per_file
                ne += ne_per_file
                if msg:
                    msgs.append(msg)
                if self.main_process:
                    pbar.desc = (
                        f'{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files'
                        )
        if self.main_process:
            pbar.close()
            with open(valid_img_record, 'w') as f:
                json.dump(cache_info, f)
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(
                f'WARNING: No labels found in {osp.dirname(img_paths[0])}. ')
    if self.task.lower() == 'val':
        if self.data_dict.get('is_coco', False):
            assert osp.exists(self.data_dict['anno_path']
                ), 'Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml'
        else:
            assert self.class_names, 'Class names is required when converting labels to coco format for evaluating.'
            save_dir = osp.join(osp.dirname(osp.dirname(img_dirs[0])),
                'annotations')
            if not osp.exists(save_dir):
                os.mkdir(save_dir)
            save_path = osp.join(save_dir, 'instances_' + osp.basename(
                img_dirs[0]) + '.json')
            TrainValDataset.generate_coco_format_labels(img_info, self.
                class_names, save_path)
    img_paths, labels = list(zip(*[(img_path, np.array(info['labels'],
        dtype=np.float32) if info['labels'] else np.zeros((0, 5), dtype=np.
        float32)) for img_path, info in img_info.items()]))
    self.img_info = img_info
    LOGGER.info(
        f'{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. '
        )
    return img_paths, labels
