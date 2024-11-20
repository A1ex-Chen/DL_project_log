def prepare_3d_data(self):
    with open(self.data_file_path_3d, 'r') as f:
        json_data = json.load(f)
    pickle_path = os.path.join(self.vision_root_path_3d,
        'scan2inst_train.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f'[!] use cache from {pickle_path} for 3d data')
    else:
        vision_embeds_3d_ref_list, vision_embeds_3d_scene_prop_list = [], []
        vision_pos_3d_ref_list, vision_pos_3d_scene_prop_list = [], []
        caption_list, task_type_list = [], []
        scene_id_list = []
        max_proposal_num = 0
        scene_id_to_3d_embeds = {}
        scene_id_to_3d_pos = {}
        scene_id_to_2d_embeds = {}
        scene_id_to_scene_scale = {}
        for scene_id in tqdm(os.listdir(os.path.join(self.
            vision_root_path_3d, 'lamm_scannet_tr3d', 'ins_pc_feat')), desc
            ='generate scene features'):
            scene_prop_feat_3d_root = os.path.join(self.vision_root_path_3d,
                'lamm_scannet_tr3d', 'ins_pc_feat', scene_id)
            obj_prop_path_list = sorted(os.listdir(scene_prop_feat_3d_root))
            scene_scale_root = os.path.join(self.vision_root_path_3d,
                'scannet_scale', scene_id + '.npy')
            scene_id_to_scene_scale[scene_id] = torch.tensor(np.load(
                scene_scale_root), dtype=torch.float32)
            max_proposal_num = max(max_proposal_num, len(obj_prop_path_list))
            scene_gt_3d_feat = []
            for obj_prop_path in obj_prop_path_list:
                scene_gt_3d_feat.append(torch.tensor(np.load(os.path.join(
                    scene_prop_feat_3d_root, obj_prop_path)), dtype=torch.
                    float16))
            scene_id_to_3d_embeds[scene_id] = torch.stack(scene_gt_3d_feat)
            scene_prop_pos_3d_root = os.path.join(self.vision_root_path_3d,
                'lamm_scannet_tr3d', 'bbox', scene_id)
            scene_prop_pos_3d = []
            for obj_prop_path in obj_prop_path_list:
                obj_prop_id, obj_prop_name = obj_prop_path.split('.')[0].split(
                    '-')
                obj_prop_bbox = np.load(os.path.join(scene_prop_pos_3d_root,
                    f'{obj_prop_id}-{obj_prop_name}.npy'))
                scene_prop_pos_3d.append(torch.tensor(np.concatenate([
                    obj_prop_bbox.min(axis=0), obj_prop_bbox.max(axis=0)]),
                    dtype=torch.float16))
            scene_id_to_3d_pos[scene_id] = torch.stack(scene_prop_pos_3d)
            scene_prop_2d_root = os.path.join(self.vision_root_path_3d,
                'instance_level_image_feat', scene_id)
            scene_prop_2d_feat = []
            for obj_prop_path in obj_prop_path_list:
                obj_prop_id, obj_prop_name = obj_prop_path.split('.')[0].split(
                    '-')
                scene_prop_2d_sub_dir = f'{obj_prop_id}_{obj_prop_name}'
                obj_prop_image_feat_path = [p for p in os.listdir(os.path.
                    join(scene_prop_2d_root, scene_prop_2d_sub_dir)) if p.
                    startswith('0_') and p.endswith('.npy')][0]
                scene_prop_2d_feat.append(torch.tensor(np.load(os.path.join
                    (scene_prop_2d_root, scene_prop_2d_sub_dir,
                    obj_prop_image_feat_path)), dtype=torch.float16).squeeze(0)
                    )
            scene_id_to_2d_embeds[scene_id] = torch.stack(scene_prop_2d_feat)
        for item in tqdm(json_data, desc='loading 3d training data'):
            task_type, caption = item.get('task_type', 'normal'), item[
                'conversations']
            caption_list.append(caption)
            task_type_list.append(task_type)
            scene_id = item['scene_id']
            scene_id_list.append(scene_id)
            vision_embeds_3d_scene_prop_list.append(scene_id_to_3d_embeds[
                item['scene_id']])
            vision_pos_3d_scene_prop_list.append(scene_id_to_3d_pos[item[
                'scene_id']])
            if task_type == 'VQA3D':
                ref_obj_name = ref_obj_id = None
                vision_embeds_3d_ref_list.append(torch.tensor(np.zeros(768),
                    dtype=torch.float16))
                vision_pos_3d_ref_list.append(torch.tensor(np.zeros(6),
                    dtype=torch.float16))
            else:
                ref_obj_name = item['object_name']
                ref_obj_id = item['object_id']
                vision_embeds_3d_ref = torch.tensor(np.load(os.path.join(
                    self.vision_root_path_3d, 'lamm_scannet_gt',
                    'ins_pc_feat', scene_id,
                    f'{ref_obj_id}-{ref_obj_name}.npy')), dtype=torch.float16)
                vision_embeds_3d_ref_list.append(vision_embeds_3d_ref.
                    reshape(-1))
                vision_pos_3d_ref = np.load(os.path.join(self.
                    vision_root_path_3d, 'lamm_scannet_gt', 'bbox',
                    scene_id, f'{ref_obj_id}-{ref_obj_name}.npy'))
                vision_pos_3d_ref = torch.tensor(np.concatenate([
                    vision_pos_3d_ref.min(axis=0), vision_pos_3d_ref.max(
                    axis=0)]), dtype=torch.float16)
                vision_pos_3d_ref_list.append(vision_pos_3d_ref.reshape(-1))
        data = {}
        data['caption_list'] = caption_list
        data['task_type_list'] = task_type_list
        data['vision_embeds_3d_ref_list'] = vision_embeds_3d_ref_list
        data['vision_embeds_3d_scene_prop_list'
            ] = vision_embeds_3d_scene_prop_list
        data['vision_pos_3d_ref_list'] = vision_pos_3d_ref_list
        data['vision_pos_3d_scene_prop_list'] = vision_pos_3d_scene_prop_list
        data['max_proposal_num'] = max_proposal_num
        data['scene_id_list'] = scene_id_list
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
    print(
        f"[!] collect {len(data['task_type_list'])} samples (loop x{self.loop_3d}) for point cloud modality training"
        )
    return data
