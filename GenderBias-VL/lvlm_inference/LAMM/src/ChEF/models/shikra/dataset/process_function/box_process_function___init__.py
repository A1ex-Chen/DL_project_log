def __init__(self, num_bins=1001):
    super().__init__()
    self.extract_box_pat = re.compile(
        '<b_st><bin_\\d*?>(?:<bin_\\d*?>){3}(?:<b_sep><bin_\\d*?>(?:<bin_\\d*?>){3})*<b_ed>'
        )
    self.extract_point_pat = re.compile(
        '<p_st><bin_\\d*?>(?:<bin_\\d*?>){1}(?:<p_sep><bin_\\d*?>(?:<bin_\\d*?>){1})*<p_ed>'
        )
    self.num_bins = num_bins
    self.use_sep = True
    self.use_begin_end = True
    self.box_begin = '<b_st>'
    self.box_sep = '<b_sep>'
    self.box_end = '<b_ed>'
    self.point_begin = '<p_st>'
    self.point_sep = '<p_sep>'
    self.point_end = '<p_ed>'
