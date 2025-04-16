def log_every(self, iterable, print_freq, header=None):
    i = 0
    if not header:
        header = ''
    start_time = time.time()
    end = time.time()
    iter_time = SmoothedValue(fmt='{avg:.4f}')
    data_time = SmoothedValue(fmt='{avg:.4f}')
    space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
    log_msg = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}',
        '{meters}', 'time: {time}', 'data: {data}']
    if torch.cuda.is_available():
        log_msg.append('max mem: {memory:.0f}')
    log_msg = self.delimiter.join(log_msg)
    MB = 1024.0 * 1024.0
    for obj in iterable:
        data_time.update(time.time() - end)
        yield obj
        iter_time.update(time.time() - end)
        if i % print_freq == 0 or i == len(iterable) - 1:
            eta_seconds = iter_time.global_avg * (len(iterable) - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if torch.cuda.is_available():
                print(log_msg.format(i, len(iterable), eta=eta_string,
                    meters=str(self), time=str(iter_time), data=str(
                    data_time), memory=torch.cuda.max_memory_allocated() / MB))
            else:
                print(log_msg.format(i, len(iterable), eta=eta_string,
                    meters=str(self), time=str(iter_time), data=str(data_time))
                    )
        i += 1
        end = time.time()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str,
        total_time / len(iterable)))
