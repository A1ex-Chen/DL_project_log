def check_updated_state(self, state: dict[str, Any]):
    if not isinstance(state, dict):
        raise ValueError('The `update` method must return a dictionary.')
    missing_types = [required_type for required_type in self.required_keys if
        required_type not in state]
    if missing_types:
        missing_types_str = ', '.join(missing_types)
        raise ValueError(
            f'The returned state from `update` is missing the following required types: {missing_types_str}.'
            )
