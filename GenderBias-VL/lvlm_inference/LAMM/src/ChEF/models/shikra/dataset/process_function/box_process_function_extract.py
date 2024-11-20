def extract(self, string: str) ->List[Boxes]:
    ret = []
    for bboxes_str in self.extract_box_pat.findall(string.replace(' ', '')):
        bboxes = []
        bbox_strs = bboxes_str.replace(self.box_begin, '').replace(self.
            box_end, '').split(self.box_sep)
        for bbox_str in bbox_strs:
            elems = list(map(int, re.findall('<bin_(\\d*?)>', bbox_str)))
            bbox = [(elem / (self.num_bins - 1)) for elem in elems]
            bboxes.append(bbox)
        ret.append(bboxes)
    return ret
