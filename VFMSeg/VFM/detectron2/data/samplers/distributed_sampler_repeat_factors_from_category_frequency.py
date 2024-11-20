@staticmethod
def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
    """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
    category_freq = defaultdict(int)
    for dataset_dict in dataset_dicts:
        cat_ids = {ann['category_id'] for ann in dataset_dict['annotations']}
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    num_images = len(dataset_dicts)
    for k, v in category_freq.items():
        category_freq[k] = v / num_images
    category_rep = {cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq)) for
        cat_id, cat_freq in category_freq.items()}
    rep_factors = []
    for dataset_dict in dataset_dicts:
        cat_ids = {ann['category_id'] for ann in dataset_dict['annotations']}
        rep_factor = max({category_rep[cat_id] for cat_id in cat_ids},
            default=1.0)
        rep_factors.append(rep_factor)
    return torch.tensor(rep_factors, dtype=torch.float32)
