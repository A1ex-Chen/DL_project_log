@classmethod
def increment_version(cls, version_list: typing.Optional[typing.List[str]]=
    None, params: typing.Optional[typing.Dict[str, str]]=None) ->str:
    version_list = version_list or []
    params = params or {}
    if params['bump_major']:
        version = cls.latest_version(version_list)
    else:
        version = cls.latest_version(version_list, major=params['major'])
    v_major, v_minor = cls._parse_version(version)
    if params['bump_major']:
        v_major += 1
        v_minor = 0
    else:
        v_minor += 1
    return f'{v_major}.{v_minor}'
