@staticmethod
def check_task(task):
    if task not in ['train', 'val', 'test', 'speed']:
        raise Exception(
            "task argument error: only support 'train' / 'val' / 'test' / 'speed' task."
            )
