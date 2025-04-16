def _convert_ids_int_string(self, slices):
    for slice in slices:
        if 'id' in slice:
            slice['id'] = str(slice['id'])
