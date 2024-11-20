def __init__(self, box_coder, anchor_generators,
    region_similarity_calculator=None, positive_fraction=None, sample_size=512
    ):
    self._region_similarity_calculator = region_similarity_calculator
    self._box_coder = box_coder
    self._anchor_generators = anchor_generators
    self._positive_fraction = positive_fraction
    self._sample_size = sample_size
