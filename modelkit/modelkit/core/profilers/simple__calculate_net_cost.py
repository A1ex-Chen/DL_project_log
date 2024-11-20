def _calculate_net_cost(self, duration: float, sub_calls: Dict[str, int]
    ) ->float:
    """Compute net cost of each sub models. Subtracting model "duration" by
        "duration" of all direct sub model.

        Args:
            duration (float): model duration
            sub_calls (Dict[str, int]): number of calls of all sub_model

        Returns:
            float: net cost
        """
    net_duration = duration
    for sub_model, num_calls in sub_calls.items():
        if sub_model == '__main__':
            continue
        if num_calls > 0:
            net_duration -= sum(self.net_durations[sub_model][-num_calls:])
    return net_duration
