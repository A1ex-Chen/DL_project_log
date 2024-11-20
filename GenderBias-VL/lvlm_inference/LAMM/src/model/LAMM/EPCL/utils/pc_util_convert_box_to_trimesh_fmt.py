def convert_box_to_trimesh_fmt(box):
    ctr = box[:3]
    lengths = box[3:]
    trns = np.eye(4)
    trns[0:3, 3] = ctr
    trns[3, 3] = 1.0
    box_trimesh_fmt = trimesh.creation.box(lengths, trns)
    return box_trimesh_fmt
