def parse_state_dict(self, state_dict):
    """Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    """
    for k, v in self.module_dict.items():
        if k in state_dict:
            v.load_state_dict(state_dict[k])
        else:
            print('Warning: Could not find %s in checkpoint!' % k)
    scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict
        }
    return scalars
