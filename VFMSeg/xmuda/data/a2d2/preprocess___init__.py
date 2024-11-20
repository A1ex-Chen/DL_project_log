def __init__(self, root_dir, scenes):
    self.class_names = A2D2Base.class_names.copy()
    self.categories = A2D2Base.categories.copy()
    self.root_dir = root_dir
    self.data = []
    self.glob_frames(scenes)
    with open(osp.join(root_dir, 'cams_lidars.json'), 'r') as f:
        self.config = json.load(f)
    with open(osp.join(root_dir, 'class_list.json'), 'r') as f:
        class_list = json.load(f)
        self.rgb_to_class = {}
        self.rgb_to_cls_idx = {}
        count = 0
        for k, v in class_list.items():
            rgb_value = tuple(int(k.lstrip('#')[i:i + 2], 16) for i in (0, 
                2, 4))
            self.rgb_to_class[rgb_value] = v
            self.rgb_to_cls_idx[rgb_value] = count
            count += 1
    assert list(class_names_to_id.keys()) == list(self.rgb_to_class.values())
