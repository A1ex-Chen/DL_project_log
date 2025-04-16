def _backward_slices(self, tp):
    res = []
    backwardTopOps = tp.query_dict(
        "select * from slices where name like '%Autograd::engine%' and depth=0 ORDER BY ts ASC"
        )
    for bto in backwardTopOps:
        kernel_slices = []
        endTime = bto['ts'] + bto['dur']
        cudaLaunchList = tp.query_dict(
            f"""select * from slices where name like '%CudaLaunchKernel%' and track_id={bto['track_id']}
                                    and depth>{bto['depth']} and ts between {bto['ts']} and {endTime}"""
            )
        for cudaLaunch in cudaLaunchList:
            slice_id_origin = cudaLaunch['slice_id']
            slice_id_destination = tp.query_dict(
                f'select * from flow where slice_out={slice_id_origin}')
            if slice_id_destination:
                cuda_slice = tp.query_dict(
                    f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}"
                    )
                kernel_slices.append(cuda_slice[0])
        bto['name'] = bto['name'].split()[-1]
        bto['kernel_list'] = kernel_slices
        res.append(bto)
    return res
