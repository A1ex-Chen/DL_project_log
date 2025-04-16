def get_raw_item(self, index=None):
    ret = copy.deepcopy({'image': self.image, 'target': {'boxes': self.
        boxes, 'points': self.points}, 'conversations': self.conversations})
    assert ret['conversations'][0]['from'] == self.roles[0]
    if ret['conversations'][-1]['from'] == self.roles[0]:
        ret['conversations'].append({'from': self.roles[1], 'value': ''})
    return ret
