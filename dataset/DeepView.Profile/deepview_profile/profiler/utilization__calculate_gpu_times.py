def _calculate_gpu_times(self, kernel_list):
    if not kernel_list:
        return 0, 0
    start, end, netTime = kernel_list[0][0], kernel_list[0][1], kernel_list[0][
        1] - kernel_list[0][0]
    for s, e in kernel_list[1:]:
        if start <= s < end and end < e:
            netTime += e - end
        elif end <= s:
            netTime += e - s
        end = max(end, e)
    return end - start, netTime
