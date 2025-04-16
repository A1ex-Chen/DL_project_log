def save_mesh_np(save_path, verts, faces=None, colors=None):
    if colors is None:
        xyz = np.hstack([np.full([verts.shape[0], 1], 'v'), verts])
        np.savetxt(save_path, xyz, fmt='%s')
    else:
        assert verts.shape[0] == colors.shape[0]
        xyzrgb = np.hstack([np.full([verts.shape[0], 1], 'v'), verts, colors])
        np.savetxt(save_path, xyzrgb, fmt='%s')
    if faces is not None:
        if faces.min == 0:
            faces += 1
        faces = faces.astype(str)
        faces = np.hstack([np.full([faces.shape[0], 1], 'f'), faces])
        with open(save_path, 'a') as f:
            np.savetxt(f, faces, fmt='%s')
