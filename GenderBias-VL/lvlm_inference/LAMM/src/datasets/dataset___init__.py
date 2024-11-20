def __init__(self, data_file_path_2d: str, data_file_path_3d: str,
    vision_root_path_2d: str, vision_root_path_3d: str, loop_2d: int,
    loop_3d: int, **kwargs):
    super().__init__()
    self.vision_type = ['image', 'pcl']
    self.data_file_path_2d = data_file_path_2d
    self.data_file_path_3d = data_file_path_3d
    self.vision_root_path_2d = vision_root_path_2d
    self.vision_root_path_3d = vision_root_path_3d
    self.loop_2d = loop_2d
    self.loop_3d = loop_3d
    self.data_2d = self.prepare_2d_data()
    self.len_2d = len(self.data_2d['vision_path_list'])
    self.data_3d = self.prepare_3d_data()
    self.len_3d = len(self.data_3d['task_type_list'])
    self.vision_type_list = ['img'] * (self.len_2d * self.loop_2d) + ['pcl'
        ] * (self.len_3d * self.loop_3d)
    self.index_list = list(range(self.len_2d)) * self.loop_2d + list(range(
        self.len_3d)) * self.loop_3d
