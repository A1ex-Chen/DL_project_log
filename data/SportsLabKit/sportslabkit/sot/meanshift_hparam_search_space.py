@property
def hparam_search_space(self):
    return {'bins': {'type': 'categorical', 'values': [8, 16, 32, 64, 128]},
        'max_iterations': {'type': 'categorical', 'values': [5, 10, 15, 20]
        }, 'termination_eps': {'type': 'categorical', 'values': [1, 2, 3, 4,
        5]}}
