@cuda.jit('(float32[:], float32[:], int32, int32, float32[:])', device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)
    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]
    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]
    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]
    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]
    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)
    if area_abc * area_abd >= 0:
        return False
    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd
    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)
    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True
