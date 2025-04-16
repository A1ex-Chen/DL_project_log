def print(self, num_recent_obs: int=1, use_colors: bool=False) ->None:
    """
        Pretty-print the Tracklet information.

        Args:
            num_recent_obs (int, optional): The number of recent observations to display for each observation type. Defaults to 1.
            use_colors (bool, optional): Whether to use colors in the output. Defaults to False.
        """
    WHITE = ''
    if use_colors:
        ENDC = '\x1b[0m'
        id_color = id_to_color(str(self.id))
    else:
        ENDC = ''
        id_color = ''
    title = (
        f'Tracklet(id={self.id}, steps_alive={self.steps_alive}, staleness={self.staleness}, is_active={self.is_active()})'
        )
    max_name_length = max([len(name) for name in self._observations.keys()])
    max_values_length = max([len(', '.join([str(val) for val in obs[-
        num_recent_obs:]])) for obs in self._observations.values()])
    box_width = max(len(title) + 4, max_name_length + max_values_length + 7)
    box_width = min(box_width, 100)
    message = f"{id_color}{'╔' + '═' * box_width + '╗'}\n"
    message += f"║ {title}{' ' * (box_width - len(title) - 1)}║\n"
    message += f"{'╟' + '─' * box_width + '╢'}{ENDC}\n"
    for name, obs in self._observations.items():
        recent_values = obs[-num_recent_obs:] if obs else []
        values_str = ', '.join([(f'{WHITE}{str(val)[:60]}{ENDC}' if len(str
            (val)) > 60 else str(val)) for val in recent_values])
        message += f'{id_color}║ {ENDC}'
        message += (
            f"{WHITE} {name}: [{values_str}]{' ' * (box_width - len(name) - len(values_str) - 6)}{ENDC}"
            )
        message += f'{id_color}║{ENDC}\n'
    message += f"{id_color}{'╚' + '═' * box_width + '╝'}{ENDC}"
    print(message)
