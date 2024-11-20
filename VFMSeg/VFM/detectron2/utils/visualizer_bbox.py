def bbox(self):
    p = mask_util.frPyObjects(self.polygons, self.height, self.width)
    p = mask_util.merge(p)
    bbox = mask_util.toBbox(p)
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox
