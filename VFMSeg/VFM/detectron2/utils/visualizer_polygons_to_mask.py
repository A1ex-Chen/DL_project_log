def polygons_to_mask(self, polygons):
    rle = mask_util.frPyObjects(polygons, self.height, self.width)
    rle = mask_util.merge(rle)
    return mask_util.decode(rle)[:, :]
