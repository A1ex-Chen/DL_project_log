def load(self, model_path, idx, c_idx=None, start_idx=0, dataset_folder=
    None, **kwargs):
    """ Loads the point cloud sequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            c_idx (int): index of category
            start_idx (int): id of sequence start
            dataset_folder (str): dataset folder
        """
    pc_seq = []
    files = self.load_files(model_path, start_idx)
    _, loc0, scale0 = self.load_single_file(files[0])
    loc_global = np.array([-0.005493, -0.1888, 0.07587]).astype(np.float32)
    scale_global = 2.338
    for f in files:
        points, loc, scale = self.load_single_file(f)
        if self.scale_type is not None:
            if self.scale_type == 'oflow':
                points = (loc + scale * points - loc0) / scale0
            if self.scale_type == 'cr':
                points = loc + scale * points
                model_id, _, frame_id = f.split('/')[-3:]
                trans = np.load(os.path.join(dataset_folder, 'smpl_params',
                    model_id, frame_id))['trans']
                points = points - trans
                if self.eval_mode:
                    points = (points - loc_global) / scale_global
        pc_seq.append(points)
    data = {None: np.stack(pc_seq), 'time': self.get_time_values()}
    if self.transform is not None:
        data = self.transform(data)
    return data
