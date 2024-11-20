def from_other(self, out_indices: Tuple[int]):
    return FeatureInfo(deepcopy(self.info), out_indices)
