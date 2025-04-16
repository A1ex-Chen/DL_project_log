def _convert_negative_tids_to_positive(self, slices):
    for slice in slices:
        if 'tid' in slice and isinstance(slice['tid'], int):
            slice['tid'] = abs(slice['tid'])
