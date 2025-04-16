@classmethod
def get_update_cli_params(cls, **kwargs) ->typing.Dict[str, typing.Any]:
    display: typing.List[str] = []
    display.append(f"Found a total of {len(kwargs['version_list'])} versions ")
    display.append(f"{kwargs['version_list']}")
    display.append(f"Last version is {kwargs['version_list'][0]}")
    return {'display': '\n'.join(display), 'params': {}}
