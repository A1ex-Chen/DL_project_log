@classmethod
def get_update_cli_params(cls, **kwargs) ->typing.Dict[str, typing.Any]:
    current_major_version = None
    if kwargs['version']:
        current_major_version, _ = cls._parse_version_str(kwargs['version'])
    major_versions = {cls._parse_version_str(v)[0] for v in kwargs[
        'version_list']}
    display = [f"Found a total of {len(kwargs['version_list'])} versions ",
        f'({len(major_versions)} major versions) ']
    for major_version in sorted(major_versions):
        display.append(f' - major `{major_version}` = ' + ', '.join(cls.
            filter_versions(kwargs['version_list'], major=str(major_version))))
    return {'display': '\n'.join(display), 'params': {'bump_major': kwargs[
        'bump_major'], 'major': current_major_version}}
