def remap(self, labelmap):
    out = np.zeros_like(labelmap, dtype=labelmap.dtype)
    for uuid in np.unique(labelmap):
        if uuid in self.label_group:
            out[labelmap == uuid] = self.label_group[uuid]
    return out
