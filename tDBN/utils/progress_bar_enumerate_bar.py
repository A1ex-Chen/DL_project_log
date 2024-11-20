def enumerate_bar(task_list, width=20, with_ptg=True, step_time_average=50,
    name=None):
    total_step = len(task_list)
    step_times = []
    start_time = 0.0
    name = '' if name is None else f'[{name}]'
    for i, task in enumerate(task_list):
        t = time.time()
        yield i, task
        step_times.append(time.time() - t)
        start_time += step_times[-1]
        start_time_str = second_to_time_str(start_time)
        average_step_time = np.mean(step_times[-step_time_average:]) + 1e-06
        speed_str = '{:.2f}it/s'.format(1 / average_step_time)
        remain_time = (total_step - i) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str((i + 1) / total_step, speed_str, time_str,
            width=width, with_ptg=with_ptg)
        print(name + prog_str + '   ', end='\r')
    print('')
