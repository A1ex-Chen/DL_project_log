def _convert_category_id(self, segment_info):
    isthing = segment_info.pop('isthing', None)
    if isthing is None:
        return segment_info
    if isthing is True:
        segment_info['category_id'] = self._thing_contiguous_id_to_dataset_id[
            segment_info['category_id']]
    else:
        segment_info['category_id'] = self._stuff_contiguous_id_to_dataset_id[
            segment_info['category_id']]
    return segment_info
