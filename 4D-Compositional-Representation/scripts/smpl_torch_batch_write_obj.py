def write_obj(self, verts, file_name):
    verts = np.around(verts, 6)
    xyz = np.hstack([np.full([verts.shape[0], 1], 'v'), verts])
    faces = np.hstack([np.full([self.faces.shape[0], 1], 'f'), self.faces + 1])
    np.savetxt(file_name, np.vstack([xyz, faces]), fmt='%s', delimiter=' ')
