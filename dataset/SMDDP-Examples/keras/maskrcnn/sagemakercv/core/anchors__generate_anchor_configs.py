def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
  Returns:
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2 ** level, scale_octave /
                    float(num_scales), aspect))
    return anchor_configs
