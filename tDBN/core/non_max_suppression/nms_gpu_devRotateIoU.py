@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def devRotateIoU(rbox1, rbox2):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    return area_inter / (area1 + area2 - area_inter)
