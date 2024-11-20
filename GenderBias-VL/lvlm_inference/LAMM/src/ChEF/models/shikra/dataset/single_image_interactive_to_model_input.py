def to_model_input(self, incontext_cfg=None):
    item = self.__getitem__(0, incontext_cfg)
    ret = {'input_text': item['input_text']}
    if 'image' in item and item['image'] is not None:
        ret['images'] = item['image'].unsqueeze(0).cuda()
    else:
        ret['images'] = None
    return ret
