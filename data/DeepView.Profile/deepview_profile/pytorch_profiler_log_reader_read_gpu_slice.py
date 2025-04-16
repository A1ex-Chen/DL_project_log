def read_gpu_slice(tp: TraceProcessor, cpu_input_slice: dict) ->dict:
    """
    Input: Perfetto handler and CPU slice
    Output: GPU kernel slice
    """
    cuda_slice = None
    slice_id_origin = cpu_input_slice['slice_id']
    slice_id_destination = tp.query_dict(
        f'select * from flow where slice_out={slice_id_origin}')
    if slice_id_destination:
        cuda_slice = tp.query_dict(
            f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}"
            )[0]
    return cuda_slice
