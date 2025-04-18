def _check_required_observations(self, target: dict[str, Any]):
    missing_types = [required_type for required_type in self.
        required_observation_types if required_type not in target]
    if missing_types:
        required_types_str = ', '.join(self.required_observation_types)
        missing_types_str = ', '.join(missing_types)
        current_types_str = ', '.join(target.keys())
        raise ValueError(
            f"""Input 'target' is missing the following required types: {missing_types_str}.
Required types: {required_types_str}
Current types in 'target': {current_types_str}"""
            )
