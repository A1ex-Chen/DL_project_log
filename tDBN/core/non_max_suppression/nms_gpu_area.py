@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i +
            4], int_pts[2 * i + 4:2 * i + 6]))
    return area_val
