def _convert_category_id(segment_info, meta):
    if segment_info['category_id'] in meta['thing_dataset_id_to_contiguous_id'
        ]:
        segment_info['category_id'] = meta['thing_dataset_id_to_contiguous_id'
            ][segment_info['category_id']]
    else:
        segment_info['category_id'] = meta['stuff_dataset_id_to_contiguous_id'
            ][segment_info['category_id']]
    return segment_info
