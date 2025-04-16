def __init__(self, target, initial_frame, bins=16, max_iterations=10,
    termination_eps=1, *args, **kwargs):
    super().__init__(target, pre_init_args={'initial_frame': initial_frame,
        'bins': bins, 'max_iterations': max_iterations, 'termination_eps':
        termination_eps})
