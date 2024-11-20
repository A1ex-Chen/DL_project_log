def export_mesh(self, mesh, model_folder, modelname, start_idx=0, n_id=1):
    """ Exports a mesh.

        Args:
            mesh (trimesh): trimesh mesh object
            model_folder (str): model folder
            modelname (str): model name
            n_id (int): number of time step (for naming convention)
        """
    colors = self.return_face_colors(len(mesh.faces))
    out_path = os.path.join(model_folder, '%s_%04d_%04d.off' % (modelname,
        start_idx, n_id))
    save_mesh(mesh, out_path, face_colors=colors)
    return out_path
