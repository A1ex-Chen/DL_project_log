def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))
    merged_lines = np.zeros((N * 8, 2))
    merged_colors = np.zeros((N * 8, 3))
    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(
        frustums):
        merged_points[i * 5:(i + 1) * 5, :] = frustum_points
        merged_lines[i * 8:(i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8:(i + 1) * 8, :] = frustum_colors
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)
    return lineset
