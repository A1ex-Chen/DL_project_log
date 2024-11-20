def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    return self.segs[key]
