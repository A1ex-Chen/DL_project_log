from torch.utils._pytree import tree_map
import torch.distributed as dist
import traceback
import orjson
from perfetto.trace_processor import TraceProcessor
import io
import torch
from typing import List, Tuple, Any
















    tp.query_dict = query_dict

    return tp


def read_gpu_slice(tp: TraceProcessor, cpu_input_slice: dict) -> dict:
    """
    Input: Perfetto handler and CPU slice
    Output: GPU kernel slice
    """
    cuda_slice = None
    slice_id_origin = cpu_input_slice["slice_id"]
    slice_id_destination = tp.query_dict(
        f"""select * from flow where slice_out={slice_id_origin}"""
    )

    if slice_id_destination:
        cuda_slice = tp.query_dict(
            f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}"
        )[0]

    return cuda_slice


def get_first_last_step(filepath: str) -> Tuple[int, int]:
    """
    Input: Pytorch profiler trace
    Output: First and last step of the trace.
    """
    tp = get_perfetto_object(filepath)
    steps_arr = tp.query_dict(f"select * from slices where name like '%ProfilerStep#%'")
    first_step = int(steps_arr[0]["name"].split("#")[1])
    last_step = int(steps_arr[-1]["name"].split("#")[1])

    return first_step, last_step


def get_ddp_forward_backward_times(filepath, step) -> Tuple[float, List[float]]:
    """
    Inputs: Pytorch profiler trace and step number
    Outputs: Forward runtime and list of ddp-bucket computation times
    """
    tp = get_perfetto_object(filepath)
    profiler_step = tp.query_dict(
        f"select * from slices where name like '%ProfilerStep#{step}%'"
    )[0]
    start_step = profiler_step["ts"]
    end_step = profiler_step["ts"] + profiler_step["dur"]
    forward_track = profiler_step["track_id"]
    backward_track = tp.query_dict(
        f"""
                                    SELECT track_id from slices
                                    WHERE name like '%autograd::%'
                                    AND ts> {start_step} and ts < {end_step}
                                    """
    )[0]["track_id"]

    ## ==================== GET TOTAL BACKWARD TIME =================== ##

    backward_slices = tp.query_dict(
        f"""
                                        select * from slices where track_id={backward_track}
                                        AND ts > {start_step} AND ts < {end_step}
                                        """
    )
    
    start_backward, end_backward = (
        backward_slices[0]["ts"],
        backward_slices[-1]["ts"] + backward_slices[-1]["dur"],
    )

    ## ==================== GET DEVICE FORWARD TIME =================== ##
    forward_slices = tp.query_dict(
        f"""
                                    select * from slices
                                    where ts > {start_step} AND ts < {start_backward}
                                    AND track_id={forward_track}
                                    """
    )
    start_forward, end_forward = (
        forward_slices[0]["ts"],
        forward_slices[-1]["ts"] + forward_slices[-1]["dur"],
    )

    forward_cuda_calls = tp.query_dict(
        f"""
                                        select * from slices where ts > {start_forward} AND ts < {end_forward}
                                        AND name like '%cudaLaunchKernel%'
                                        """
    )

    forward_first_cuda_call = forward_cuda_calls[0]
    forward_last_cuda_call = forward_cuda_calls[-1]
    forward_device_start = 0
    forward_device_ends = 0
    if forward_first_cuda_call:
        cuda_slice_start = read_gpu_slice(tp, forward_first_cuda_call)
        forward_device_start = cuda_slice_start["ts"] if cuda_slice_start else 0
    if forward_last_cuda_call:
        cuda_slice_end = read_gpu_slice(tp, forward_last_cuda_call)
        forward_device_ends = (
            cuda_slice_end["ts"] + cuda_slice_end["dur"] if cuda_slice_end else 0
        )

    forward_start_ts = (
        forward_device_start if forward_device_start != 0 else start_forward
    )
    forward_end_ts = forward_device_ends if forward_device_ends != 0 else end_forward

    ## ==================== GET BUCKET TIMES =================== ##
    find_c10_all_reduce_calls = f"""
                                SELECT * from slice main
                                WHERE main.track_id={backward_track}
                                AND main.ts > {start_backward} AND main.ts < {end_backward}
                                AND main.name like '%autograd::engine::evaluate_function: torch::autograd::AccumulateGrad%'
                                AND 'c10d::allreduce_' IN (SELECT submain.name FROM slice submain WHERE submain.ts > main.ts AND submain.ts < main.ts + main.dur AND submain.track_id={backward_track} )    
                                """
    all_reduce_calls = tp.query_dict(find_c10_all_reduce_calls)
    prev_ts = start_step
    bucket_comp_times = []
    for idx, item in enumerate(all_reduce_calls):
        # ================== COMPUTATION ======================================#
        end_ts = item["ts"]
        slices_in_bucket = tp.query_dict(
            f"""
                                         select * from slice
                                         where track_id={backward_track}
                                         and ts > {prev_ts} and ts < {end_ts}                
                                        """
        )
        start_cpu_time = slices_in_bucket[0]["ts"]
        end_cpu_time = slices_in_bucket[-1]["ts"] + slices_in_bucket[-1]["dur"]

        device_ts_start = 0
        device_ts_end = 0

        cuda_launch_list = tp.query_dict(
            f"""
                                        select * from slice
                                        where ts > {start_cpu_time} and ts < {end_cpu_time} and track_id={backward_track}
                                        and name like '%cudaLaunchKernel%'
                                        """
        )
        if cuda_launch_list:
            cuda_slice_start = read_gpu_slice(tp, cuda_launch_list[0])
            device_ts_start = cuda_slice_start["ts"] if cuda_slice_start else 0

            cuda_slice_end = read_gpu_slice(tp, cuda_launch_list[-1])
            device_ts_end = (
                cuda_slice_end["ts"] + cuda_slice_end["dur"] if cuda_slice_end else 0
            )

        netCompTime = max(0, device_ts_end - device_ts_start)

        bucket_comp_times.append(round(netCompTime * 1e-6, 3))

        prev_ts = item["ts"] + item["dur"]

    forward_time = round((forward_end_ts - forward_start_ts) * 1e-6, 3)

    return forward_time, bucket_comp_times