def evaluate(self):
    if self.distributed:
        comm.synchronize()
        all_time = comm.gather(self.all_time, dst=0)
        all_time = list(itertools.chain(*all_time))
        if not comm.is_main_process():
            return {}
    else:
        all_time = self.all_time
    if len(all_time) == 0:
        return {'GPU_Speed': 0}
    all_time = np.array(all_time)
    speeds = 1.0 / all_time
    if self.unit == 'minisecond':
        speeds *= 1000
    mean_speed = speeds.mean()
    std_speed = speeds.std()
    max_speed = speeds.max()
    min_speed = speeds.min()
    mid_speed = np.median(speeds)
    if self.out_file is not None:
        f = open(self.out_file, 'a')
        curr_time = time.strftime('%Y/%m/%d,%H:%M:%S', time.localtime())
        f.write('%s\t%.2f\n' % (curr_time, mean_speed))
        f.close()
    ret_dict = {'Mean_FPS': mean_speed, 'Std_FPS': std_speed, 'Max_FPS':
        max_speed, 'Min_FPS': min_speed, 'Mid_FPS': mid_speed}
    return {'GPU_Speed': ret_dict}
