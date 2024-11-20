def _calculate_gpu_forward_time(self, tp, node):
    span = 0
    time = 0
    cudaLaunchList = tp.query_dict(
        f"""select * from slices where name like '%CudaLaunchKernel%' and track_id={node.track}
                                    and depth>{node.depth} and ts between {node.start} and {node.end} ORDER BY ts ASC"""
        )
    kernel_list = []
    for cudaLaunch in cudaLaunchList:
        slice_id_origin = cudaLaunch['slice_id']
        slice_id_destination = tp.query_dict(
            f'select * from flow where slice_out={slice_id_origin}')
        if slice_id_destination:
            cuda_slice = tp.query_dict(
                f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}"
                )
            kernel_list.append((cuda_slice[0]['ts'], cuda_slice[0]['ts'] +
                cuda_slice[0]['dur']))
    if kernel_list:
        kernel_list.sort(key=lambda x: x[0])
        span, time = self._calculate_gpu_times(kernel_list)
    node.gpu_forward_span = span
    node.gpu_forward = time
