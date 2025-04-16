def extract_point(self, string: str) ->List[Boxes]:
    ret = []
    for bboxes_str in self.extract_point_pat.findall(string):
        bboxes = []
        bbox_strs = bboxes_str.replace(self.point_begin, '').replace(self.
            point_end, '').split(self.point_sep)
        for bbox_str in bbox_strs:
            elems = list(map(int, re.findall('<bin_(\\d*?)>', bbox_str)))
            bbox = [(elem / (self.num_bins - 1)) for elem in elems]
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret
