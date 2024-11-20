def preprcess_nuScene(SEEM, SAM, dataset='', scene='', Resize=None):
    mapping = 'NuScenesLidarSegSCN'
    pkl_data = []
    curr_split = dataset
    print('load pkl data...')
    if 'USA_SING' == scene:
        save_path = save_new_usa_singapore_pkl_path
        if 'train' == curr_split:
            pkl_path_list = [pkl_path_train_usa_path,
                pkl_path_train_singapore_path]
            if Resize:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split +
                    '_resize', save_path + '/' + 'trg' + '/' + curr_split +
                    '_resize']
            else:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split,
                    save_path + '/' + 'trg' + '/' + curr_split]
        elif 'test' == curr_split:
            pkl_path_list = [pkl_path_test_usa_path,
                pkl_path_test_singapore_path]
            save_dir_list = [save_path + '/' + 'src' + '/' + curr_split, 
                save_path + '/' + 'trg' + '/' + curr_split]
        elif 'val' == curr_split:
            pkl_path_list = [pkl_path_val_singapore_path]
            save_dir_list = [save_path + '/' + curr_split]
        else:
            return
    elif 'DAY_NIGHT' == scene:
        save_path = save_new_day_night_pkl_path
        if 'train' == curr_split:
            pkl_path_list = [pkl_path_train_day_path, pkl_path_train_night_path
                ]
            if Resize:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split +
                    '_resize', save_path + '/' + 'trg' + '/' + curr_split +
                    '_resize']
            else:
                save_dir_list = [save_path + '/' + 'src' + '/' + curr_split,
                    save_path + '/' + 'trg' + '/' + curr_split]
        elif 'test' == curr_split:
            pkl_path_list = [pkl_path_test_day_path, pkl_path_test_night_path]
            save_dir_list = [save_path + '/' + 'src' + '/' + curr_split, 
                save_path + '/' + 'trg' + '/' + curr_split]
        elif 'val' == curr_split:
            pkl_path_list = [pkl_path_val_night_path]
            save_dir_list = [save_path + '/' + curr_split]
        else:
            return
    else:
        return
    for pkl_path, save_dir in zip(pkl_path_list, save_dir_list):
        print('process ' + scene + ' ' + curr_split + '...')
        os.makedirs(save_dir, exist_ok=True)
        with open(pkl_path, 'rb') as f:
            pkl_data.extend(pickle.load(f))
        print('iterate pkl data...')
        pkl_files = []
        accu_id = 0
        with tqdm(total=len(pkl_data)) as bar:
            for pkl_id, data in enumerate(pkl_data):
                new_pkl = {}
                img_path = osp.join(nuScene_orig_data_path, data['camera_path']
                    )
                image = Image.open(img_path)
                if Resize:
                    image = image.resize(Resize, Image.BILINEAR)
                sam_masks = SAM.generate(cv2.cvtColor(np.array(image).
                    astype(np.uint8), cv2.COLOR_BGR2RGB))
                seem_masks, _ = call_SEEM(SEEM, pil_image=image, mapping=
                    mapping)
                sam_masks = [x['segmentation'] for x in sam_masks]
                sam_masks = merge_sam_masks(sam_masks)
                seem_masks = [x for x in seem_masks.cpu().numpy()]
                new_pkl['sam'] = sam_masks.copy()
                new_pkl['seem'] = seem_masks.copy()
                pkl_files.append(new_pkl)
                if 0 == len(pkl_files) % 10:
                    for pkl in pkl_files:
                        pkl_name = '{}.pkl'.format(str(accu_id))
                        with open(osp.join(save_dir, pkl_name), 'wb') as f:
                            pickle.dump([pkl], f)
                        accu_id += 1
                    pkl_files = []
                bar.update(1)
        if 0 != len(pkl_files):
            for pkl in pkl_files:
                pkl_name = '{}.pkl'.format(str(accu_id))
                with open(osp.join(save_dir, pkl_name), 'wb') as f:
                    pickle.dump([pkl], f)
                accu_id += 1
            pkl_files = []
    return
