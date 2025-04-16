def draw_3d_bbox_meshes_in_pyqt(widget, bboxes, colors=GLColor.Gray, alpha=
    1.0, edgecolors=None):
    verts, faces = _3d_bbox_to_mesh(bboxes)
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = np.array([colors for i in range(len(verts))])
    m1 = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors,
        smooth=False)
    m1.setGLOptions('additive')
    widget.addItem(m1)
    return widget
