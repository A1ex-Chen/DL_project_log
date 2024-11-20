def return_face_colors(self, n_faces):
    """ Returns the face colors.

        Args:
            n_faces (int): number of faces
        """
    if self.mesh_color:
        step_size = 255.0 / n_faces
        colors = [[int(255 - i * step_size), 25, int(i * step_size), 255] for
            i in range(n_faces)]
        colors = np.array(colors).astype(np.uint64)
    else:
        colors = None
    return colors
