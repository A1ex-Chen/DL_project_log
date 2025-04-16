def draw_absolute_grid(frame: np.ndarray, coord_transformations:
    CoordinatesTransformation, grid_size: int=20, radius: int=2, thickness:
    int=1, color: ColorType=Color.black, polar: bool=False):
    """
    Draw a grid of points in absolute coordinates.

    Useful for debugging camera motion.

    The points are drawn as if the camera were in the center of a sphere and points are drawn in the intersection
    of latitude and longitude lines over the surface of the sphere.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on.
    coord_transformations : CoordinatesTransformation
        The coordinate transformation as returned by the [`MotionEstimator`][norfair.camera_motion.MotionEstimator]
    grid_size : int, optional
        How many points to draw.
    radius : int, optional
        Size of each point.
    thickness : int, optional
        Thickness of each point
    color : ColorType, optional
        Color of the points.
    polar : Bool, optional
        If True, the points on the first frame are drawn as if the camera were pointing to a pole (viewed from the center of the earth).
        By default, False is used which means the points are drawn as if the camera were pointing to the Equator.
    """
    h, w, _ = frame.shape
    points = _get_grid(grid_size, w, h, polar=polar)
    if coord_transformations is None:
        points_transformed = points
    else:
        points_transformed = coord_transformations.abs_to_rel(points)
    visible_points = points_transformed[(points_transformed <= np.array([w,
        h])).all(axis=1) & (points_transformed >= 0).all(axis=1)]
    for point in visible_points:
        Drawer.cross(frame, point.astype(int), radius=radius, thickness=
            thickness, color=color)
