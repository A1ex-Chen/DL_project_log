def export(self, meshes, mesh_dir, modelname, start_idx=0, start_id_seq=1):
    """ Exports a list of meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        """
    model_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    return self.export_meshes_t(meshes, model_folder, modelname, start_idx=
        0, start_id_seq=1)
