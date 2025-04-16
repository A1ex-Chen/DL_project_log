def format_box(self, bboxes: Boxes) ->str:
    final_str = []
    for bbox in bboxes:
        quant_x0 = '<bin_{}>'.format(round(bbox[0] * (self.num_bins - 1)))
        quant_y0 = '<bin_{}>'.format(round(bbox[1] * (self.num_bins - 1)))
        quant_x1 = '<bin_{}>'.format(round(bbox[2] * (self.num_bins - 1)))
        quant_y1 = '<bin_{}>'.format(round(bbox[3] * (self.num_bins - 1)))
        region_coord = '{} {} {} {}'.format(quant_x0, quant_y0, quant_x1,
            quant_y1)
        final_str.append(region_coord)
    if self.use_sep:
        final_str = self.box_sep.join(final_str)
    else:
        final_str = ''.join(final_str)
    if self.use_begin_end:
        final_str = self.box_begin + final_str + self.box_end
    return final_str
