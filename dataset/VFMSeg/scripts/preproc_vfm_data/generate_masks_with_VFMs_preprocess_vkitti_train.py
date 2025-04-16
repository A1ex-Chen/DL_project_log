def preprocess_vkitti_train(SEEM, SAM):
    mapping = 'SemanticKITTISCN'
    curr_split = 'train'
    pkl_data = []
    print('load pkl data...')
    with open(pkl_path_VKITTI_train, 'rb') as f:
        pkl_data.extend(pickle.load(f))
    print('iterate pkl data...')
    save_dir = save_new_vkitti_pkl_path + '/' + curr_split
    os.makedirs(save_dir, exist_ok=True)
    random_weather = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    pkl_files = []
    accu_id = 0
    try:
        with tqdm(total=len(pkl_data) * 6) as bar:
            for pkl_id, data in enumerate(pkl_data):
                new_pkl = {}
                images = {}
                for weather in random_weather:
                    img_path = osp.join(vkitti_orig_data_path,
                        'vkitti_1.3.1_rgb', data['scene_id'], weather, data
                        ['frame_id'] + '.png')
                    images[weather] = Image.open(img_path)
                image_sam_masks = {}
                image_seem_masks = {}
                for idx, key in enumerate(images):
                    sam_masks = SAM.generate(cv2.cvtColor(np.array(images[
                        key]).astype(np.uint8), cv2.COLOR_BGR2RGB))
                    seem_masks, _ = call_SEEM(SEEM, pil_image=images[key],
                        mapping=mapping)
                    sam_masks = [x['segmentation'] for x in sam_masks]
                    sam_masks = merge_sam_masks(sam_masks)
                    seem_masks = [x for x in seem_masks.cpu().numpy()]
                    image_sam_masks[key] = sam_masks
                    image_seem_masks[key] = seem_masks
                    bar.update(1)
                new_pkl['sam'] = image_sam_masks.copy()
                new_pkl['seem'] = image_seem_masks.copy()
                pkl_files.append(new_pkl)
                if 0 == len(pkl_files) % 5:
                    for pkl in pkl_files:
                        pkl_name = '{}.pkl'.format(str(accu_id))
                        with open(osp.join(save_dir, pkl_name), 'wb') as f:
                            pickle.dump([pkl], f)
                        accu_id += 1
                    pkl_files = []
        if 0 != len(pkl_files):
            for pkl in pkl_files:
                pkl_name = '{}.pkl'.format(str(accu_id))
                with open(osp.join(save_dir, pkl_name), 'wb') as f:
                    pickle.dump([pkl], f)
                accu_id += 1
            pkl_files = []
    except:
        print('Exception Occured!')
    else:
        pass
    return
