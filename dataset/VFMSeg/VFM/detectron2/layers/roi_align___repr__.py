def __repr__(self):
    tmpstr = self.__class__.__name__ + '('
    tmpstr += 'output_size=' + str(self.output_size)
    tmpstr += ', spatial_scale=' + str(self.spatial_scale)
    tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
    tmpstr += ', aligned=' + str(self.aligned)
    tmpstr += ')'
    return tmpstr
