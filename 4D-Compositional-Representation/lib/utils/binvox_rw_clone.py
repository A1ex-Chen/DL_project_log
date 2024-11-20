def clone(self):
    data = self.data.copy()
    dims = self.dims[:]
    translate = self.translate[:]
    return Voxels(data, dims, translate, self.scale, self.axis_order)
