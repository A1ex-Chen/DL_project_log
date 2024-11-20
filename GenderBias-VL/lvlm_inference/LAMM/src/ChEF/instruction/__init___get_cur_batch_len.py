def get_cur_batch_len(batch):
    if 'id' in batch:
        cur_batch_len = len(batch['id'])
    elif 'image_path' in batch:
        cur_batch_len = len(batch['image_path'])
    elif 'pcl_paths' in batch:
        cur_batch_len = len(batch['pcl_paths'])
    elif 'task_type' in batch:
        cur_batch_len = len(batch['task_type'])
    else:
        raise ValueError('cannot get batch size')
    return cur_batch_len
