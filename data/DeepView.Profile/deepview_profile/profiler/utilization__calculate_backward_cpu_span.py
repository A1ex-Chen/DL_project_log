def _calculate_backward_cpu_span(self, node):
    backward_slices = node.cpu_backward_slices
    kernel_slices = []
    if not backward_slices:
        return
    minTs = float('inf')
    maxTs = float('-inf')
    netTime = 0
    backward_slices.sort(key=lambda x: x['ts'])
    for bw_slice in backward_slices:
        minTs = min(minTs, bw_slice['ts'])
        maxTs = max(maxTs, bw_slice['ts'] + bw_slice['dur'])
        netTime += bw_slice['dur']
        kernel_slices.extend([(kernel['ts'], kernel['ts'] + kernel['dur']) for
            kernel in bw_slice['kernel_list']])
    node.cpu_backward_span = maxTs - minTs
    node.cpu_backward = netTime
    if kernel_slices:
        kernel_slices.sort(key=lambda x: x[0])
        node.gpu_backward_span, node.gpu_backward = self._calculate_gpu_times(
            kernel_slices)
