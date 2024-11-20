def export(self, meshes, mesh_dir, modelname, start_idx=0, start_id_seq=1):
    """ Exports a list of meshes.

        Args:
            meshes (list): list of trimesh meshes
            mesh_dir (str): mesh directory
            modelname (str): model name
            start_idx (int): start id of sequence (for naming convention)
            start_id_seq (int): id of start mesh in its sequence
                (e.g. 1 for start mesh)
        """
    model_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    return self.export_multiple_meshes(meshes, model_folder, modelname,
        start_idx, start_id_seq=start_id_seq)
