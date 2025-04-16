def export_mesh(self, mesh, model_folder, modelname, start_idx=0, n_id=1,
    out_format='off'):
    """ Exports a mesh.

        Args:
            mesh(trimesh): mesh to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            n_id (int): number of mesh in the sequence (e.g. 1 -> start)
        """
    if out_format == 'obj':
        out_path = os.path.join(model_folder, '%s_%04d_%04d.obj' % (
            modelname, start_idx, n_id))
        export_mesh(mesh, out_path)
    else:
        out_path = os.path.join(model_folder, '%s_%04d_%04d.off' % (
            modelname, start_idx, n_id))
        save_mesh(mesh, out_path)
    return out_path
