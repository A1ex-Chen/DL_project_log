def flatten(self):
    ret = []
    for _, v in self.batch_extra_fields.items():
        if isinstance(v, (Boxes, Keypoints)):
            ret.append(v.tensor)
        else:
            ret.append(v)
    return ret
