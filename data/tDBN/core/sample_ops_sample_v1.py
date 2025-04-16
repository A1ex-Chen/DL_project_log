def sample_v1(self, name, num):
    if isinstance(name, (list, tuple)):
        group_name = ', '.join(name)
        ret = self._sampler_dict[group_name].sample(num)
        groups_num = [len(l) for l in ret]
        return reduce(lambda x, y: x + y, ret), groups_num
    else:
        ret = self._sampler_dict[name].sample(num)
        return ret, np.ones((len(ret),), dtype=np.int64)
