@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)
    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)
    num_intersection = quadrilateral_intersection(corners1, corners2,
        intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return area(intersection_corners, num_intersection)
