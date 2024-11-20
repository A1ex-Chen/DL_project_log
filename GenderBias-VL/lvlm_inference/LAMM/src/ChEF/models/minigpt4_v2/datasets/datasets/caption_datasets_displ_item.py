def displ_item(self, index):
    sample, ann = self.__getitem__(index), self.annotation[index]
    return OrderedDict({'file': ann['image'], 'caption': ann['caption'],
        'image': sample['image']})
