def _3d_bbox_to_mesh(bboxes):
    bbox_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 
        4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2], [3, 2, 6], [3, 6, 7], [0, 1,
        5], [0, 5, 4]])
    verts_list = []
    faces_list = []
    for i, bbox in enumerate(bboxes):
        verts_list.append(bbox)
        faces_list.append(bbox_faces + 8 * i)
    verts = np.concatenate(verts_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    return verts, faces
