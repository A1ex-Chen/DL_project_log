@staticmethod
def _reset_ckpt_args(args: dict) ->dict:
    """Reset arguments when loading a PyTorch model."""
    include = {'imgsz', 'data', 'task', 'single_cls'}
    return {k: v for k, v in args.items() if k in include}
