def __getitem__(self, i):
    vision_type = self.vision_type_list[i]
    index = self.index_list[i]
    if vision_type == 'img':
        return self.get_2d_data(index)
    else:
        return self.get_3d_data(index)
