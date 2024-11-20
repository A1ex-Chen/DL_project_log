def export_meshes_t(self, meshes, model_folder, modelname, start_idx=0,
    start_id_seq=2):
    """ Exports meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        """
    out_files = []
    for i, m in enumerate(meshes):
        out_file = self.export_mesh(m, model_folder, modelname, start_idx,
            n_id=start_id_seq + i)
        out_files.append(out_file)
    return out_files
