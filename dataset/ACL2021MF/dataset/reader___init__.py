def __init__(self, boxes_h5path: str, detection_dict: Dict[int, str],
    object_blacklist_path: str, class_structure_path: str=None,
    abstract_list_path: str=None, min_score: float=0.01, top_k: int=3,
    is_val: bool=False, cls_start_index: int=0, object_filtering: bool=True,
    variant_copy_candidates: bool=True, in_memory: bool=False,
    copy_candidate_clear_up=False):
    with open(object_blacklist_path) as out:
        blacklist = json.load(out)
        full_list = blacklist['blacklist_categories'] + (blacklist[
            'val_blacklist_categories'] if is_val else [])
    self._blacklist_categories = set([s.lower() for s in full_list])
    self._boxes_h5path = boxes_h5path
    self.detection_dict = detection_dict
    self.min_score = min_score
    self.is_val = is_val
    self.object_filtering = object_filtering
    self.top_k = top_k
    self.cls_start_index = cls_start_index
    self.in_memory = in_memory
    self.variant_copy_candidates = variant_copy_candidates
    self.copy_candidate_clear_up = copy_candidate_clear_up
    if abstract_list_path is not None and class_structure_path is not None:
        self.hierarchy_finder = HierarchyFinder(class_structure_path,
            abstract_list_path)
    else:
        self.hierarchy_finder = None
    self.cache = {}
    with h5py.File(self._boxes_h5path, 'r') as boxes_h5:
        self._width = boxes_h5['width'][:]
        self._height = boxes_h5['height'][:]
        self._image_ids = boxes_h5['image_id'][:].tolist()
        self._image_ids = {image_id: index for index, image_id in enumerate
            (self._image_ids)}
        if self.in_memory:
            for image_id in tqdm(self._image_ids):
                self.process_single_image(image_id, self._image_ids[
                    image_id], boxes_h5)
