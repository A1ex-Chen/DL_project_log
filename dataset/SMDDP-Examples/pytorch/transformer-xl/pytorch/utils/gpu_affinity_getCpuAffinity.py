def getCpuAffinity(self):
    affinity_string = ''
    for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, device.
        _nvml_affinity_elements):
        affinity_string = '{:064b}'.format(j) + affinity_string
    affinity_list = [int(x) for x in affinity_string]
    affinity_list.reverse()
    ret = [i for i, e in enumerate(affinity_list) if e != 0]
    return ret
