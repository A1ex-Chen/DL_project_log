def summary(self, *args, print_table: bool=False, **kwargs) ->Union[Dict[
    str, List], str]:
    """Usage

            stat: Dict[str, List] = profiler.summary()
        or
            print(profiler.summary(print_table=True, tablefmt="fancy_grid"))

        See: https://pypi.org/project/tabulate/ for all available table formats.

        """
    result = defaultdict(list)
    total = self.total_duration
    for model_name in self.durations:
        durations = self.durations[model_name]
        result['Name'].append(model_name)
        result['Duration per call (s)'].append(sum(durations) / len(
            durations) if durations else 0)
        result['Num call'].append(len(durations))
        result['Total duration (s)'].append(sum(durations))
        result['Total percentage %'].append(100 * sum(durations) / total)
        net_durations = self.net_durations[model_name]
        result['Net duration per call (s)'].append(sum(net_durations) / len
            (net_durations) if net_durations else 0)
    result['Net percentage %'] = [(100 * net * num / total) for net, num in
        zip(result['Net duration per call (s)'], result['Num call'])]
    arg_index = sorted(range(len(result['Name'])), key=result[
        'Total percentage %'].__getitem__, reverse=True)
    result_sorted = OrderedDict()
    for col in ['Name', 'Net duration per call (s)', 'Net percentage %',
        'Num call', 'Duration per call (s)', 'Total duration (s)',
        'Total percentage %']:
        result_sorted[col] = [*map(lambda x: result[col][x], arg_index)]
    if print_table:
        return tabulate(result_sorted, headers='keys', **kwargs)
    return result_sorted
