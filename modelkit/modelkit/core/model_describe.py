def describe(self, t=None):
    if not t:
        t = Tree('')
    if self.configuration_key:
        sub_t = t.add(
            f'[deep_sky_blue1]configuration[/deep_sky_blue1]: [orange3]{self.configuration_key}'
            )
    if self.__doc__:
        t.add(f'[deep_sky_blue1]doc[/deep_sky_blue1]: {self.__doc__.strip()}')
    if self._item_type and self._return_type:
        sub_t = t.add(
            f'[deep_sky_blue1]signature[/deep_sky_blue1]: {pretty_print_type(self._item_type)} -> {pretty_print_type(self._return_type)}'
            )
    if self._load_time:
        sub_t = t.add(
            '[deep_sky_blue1]load time[/deep_sky_blue1]: [orange3]' +
            humanize.naturaldelta(dt.timedelta(seconds=self._load_time),
            minimum_unit='milliseconds'))
    if self._load_memory_increment is not None:
        sub_t = t.add(
            f'[deep_sky_blue1]load memory[/deep_sky_blue1]: [orange3]{humanize.naturalsize(self._load_memory_increment)}'
            )
    if self.asset_path:
        sub_t = t.add(
            f'[deep_sky_blue1]asset path[/deep_sky_blue1]: [orange3]{self.asset_path}'
            )
    if self.batch_size:
        sub_t = t.add(
            f'[deep_sky_blue1]batch size[/deep_sky_blue1]: [orange3]{self.batch_size}'
            )
    if self.model_settings:
        sub_t = t.add('[deep_sky_blue1]model settings[/deep_sky_blue1]')
        describe(self.model_settings, t=sub_t)
    if self.model_dependencies.models:
        dep_t = t.add('[deep_sky_blue1]dependencies')
        for m in self.model_dependencies.models:
            dep_t.add('[orange3]' + escape(m))
        global_load_time, global_load_memory = (self.
            _compute_dependencies_load_info())
        sub_t = t.add(
            '[deep_sky_blue1]load time including dependencies[/deep_sky_blue1]:'
             + ' [orange3]' + humanize.naturaldelta(dt.timedelta(seconds=
            global_load_time), minimum_unit='milliseconds'))
        sub_t = t.add(
            '[deep_sky_blue1]load memory including dependencies[/deep_sky_blue1]:'
             + ' [orange3]' + humanize.naturalsize(global_load_memory))
    return t
