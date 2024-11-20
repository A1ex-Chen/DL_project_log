def encode_json_sem_seg(self, sem_seg, input_file_name):
    """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
    json_list = []
    for label in np.unique(sem_seg):
        if self._contiguous_id_to_dataset_id is not None:
            assert label in self._contiguous_id_to_dataset_id, 'Label {} is not in the metadata info for {}'.format(
                label, self._dataset_name)
            dataset_id = self._contiguous_id_to_dataset_id[label]
        else:
            dataset_id = int(label)
        mask = (sem_seg == label).astype(np.uint8)
        mask_rle = mask_util.encode(np.array(mask[:, :, None], order='F'))[0]
        mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
        json_list.append({'file_name': input_file_name, 'category_id':
            dataset_id, 'segmentation': mask_rle})
    return json_list
