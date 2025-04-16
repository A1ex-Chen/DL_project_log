def load_all_steps(self, files, loc0, scale0, loc_global, scale_global,
    dataset_folder):
    """ Loads data for all steps.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        """
    p_list = []
    o_list = []
    t_list = []
    for i, f in enumerate(files):
        points_dict = np.load(f)
        points = points_dict['points']
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 0.0001 * np.random.randn(*points.shape)
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)
        loc = points_dict['loc'].astype(np.float32)
        scale = points_dict['scale'].astype(np.float32)
        model_id, _, frame_id = f.split('/')[-3:]
        if self.spatial_completion:
            data_folder = os.path.join(dataset_folder, 'test', 'D-FAUST',
                model_id)
            mask_folder = os.path.join(dataset_folder, 'spatial_mask', model_id
                )
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            mask_file = os.path.join(mask_folder, frame_id.replace('.npz',
                '.npy'))
            if os.path.exists(mask_file):
                mask = np.load(mask_file)
            else:
                pcl = np.load(os.path.join(data_folder, 'pcl_seq', frame_id))[
                    'points']
                mask, _, _ = random_crop_occ(points, pcl)
                np.save(mask_file, mask)
            points = points[mask, :]
            occupancies = occupancies[mask]
        if self.scale_type is not None:
            if self.scale_type == 'oflow':
                points = (loc + scale * points - loc0) / scale0
            if self.scale_type == 'cr':
                trans = np.load(os.path.join(dataset_folder, 'smpl_params',
                    model_id, frame_id))['trans']
                loc -= trans
                points = (loc + scale * points - loc_global) / scale_global
        points = points.astype(np.float32)
        time = np.array(i / (self.seq_len - 1), dtype=np.float32)
        p_list.append(points)
        o_list.append(occupancies)
        t_list.append(time)
    if not self.spatial_completion:
        data = {None: np.stack(p_list), 'occ': np.stack(o_list), 'time': np
            .stack(t_list)}
    else:
        data = {None: p_list, 'occ': o_list, 'time': np.stack(t_list)}
    return data
