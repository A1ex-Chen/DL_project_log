def preprocess_skitti(SEEM, SAM, dataset='', mapping='SemanticKITTISCN'):
    mapping = mapping
    pkl_data = []
    curr_split = dataset
    print('load pkl data...')
    if 'train' == dataset:
        pkl_path = pkl_path_SKITTI_train
    elif 'val' == dataset:
        pkl_path = pkl_path_SKITTI_val
    elif 'test' == dataset:
        pkl_path = pkl_path_SKITTI_test
    else:
        return
    if mapping == 'SemanticKITTISCN':
        save_dir = save_new_skitti_pkl_path + '/' + curr_split
    else:
        save_dir = save_new_skitti_pkl_path + '/' + curr_split + '_for_a2d2'
    os.makedirs(save_dir, exist_ok=True)
    with open(pkl_path, 'rb') as f:
        pkl_data.extend(pickle.load(f))
    print('iterate pkl data...')
    pkl_files = []
    accu_id = 0
    with tqdm(total=len(pkl_data)) as bar:
        for pkl_id, data in enumerate(pkl_data):
            new_pkl = {}
            img_path = osp.join(skitti_orig_data_path, data['camera_path'])
            image = Image.open(img_path)
            sam_masks = SAM.generate(cv2.cvtColor(np.array(image).astype(np
                .uint8), cv2.COLOR_BGR2RGB))
            seem_masks, _ = call_SEEM(SEEM, pil_image=image, mapping=mapping)
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
    return
