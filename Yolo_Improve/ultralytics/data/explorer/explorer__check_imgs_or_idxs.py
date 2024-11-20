def _check_imgs_or_idxs(self, img: Union[str, np.ndarray, List[str], List[
    np.ndarray], None], idx: Union[None, int, List[int]]) ->List[np.ndarray]:
    """Determines whether to fetch images or indexes based on provided arguments and returns image paths."""
    if img is None and idx is None:
        raise ValueError('Either img or idx must be provided.')
    if img is not None and idx is not None:
        raise ValueError('Only one of img or idx must be provided.')
    if idx is not None:
        idx = idx if isinstance(idx, list) else [idx]
        img = self.table.to_lance().take(idx, columns=['im_file']).to_pydict()[
            'im_file']
    return img if isinstance(img, list) else [img]
