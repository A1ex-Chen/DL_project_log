def format_point(self, points) ->str:
    final_str = []
    for bbox in points:
        quant_x0 = '<bin_{}>'.format(round(bbox[0] * (self.num_bins - 1)))
        quant_y0 = '<bin_{}>'.format(round(bbox[1] * (self.num_bins - 1)))
        region_coord = '{} {}'.format(quant_x0, quant_y0)
        final_str.append(region_coord)
    if self.use_sep:
        final_str = self.point_sep.join(final_str)
    else:
        final_str = ''.join(final_str)
    if self.use_begin_end:
        final_str = self.point_begin + final_str + self.point_end
    return final_str
