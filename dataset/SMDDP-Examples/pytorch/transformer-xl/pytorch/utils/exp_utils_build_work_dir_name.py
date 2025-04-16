def build_work_dir_name(work_dir, dataset, append_dataset, append_time):
    if append_dataset:
        work_dir = '{}-{}'.format(work_dir, dataset)
    if append_time:
        now = int(time.time())
        now_max = utils.distributed.all_reduce_item(now, op='max')
        now_str = datetime.datetime.fromtimestamp(now_max).strftime(
            '%Y%m%d-%H%M%S')
        work_dir = os.path.join(work_dir, now_str)
    return work_dir
